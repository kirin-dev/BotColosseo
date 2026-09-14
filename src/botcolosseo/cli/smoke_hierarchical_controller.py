"""Live plumbing check with untrained strategic actors; never a skill result."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from botcolosseo.agents.hierarchical_controller import HierarchicalController
from botcolosseo.agents.hierarchical_critic import StrategicActorCritic
from botcolosseo.agents.hierarchical_model import CommandExecutor, StrategicActor
from botcolosseo.data.extraction_demonstrations import extraction_scalars
from botcolosseo.data.hierarchical_demonstrations import load_command_episode
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv
from botcolosseo.training.extraction_rollout import extraction_privileged_tensor
from botcolosseo.training.hierarchical_protocol import COMMAND_SCHEMA, ControlCondition, own_payoffs
from botcolosseo.training.hierarchical_smdp import aggregate_strategic_transitions
from botcolosseo.training.hierarchical_strategic_ppo import HIGH_INPUT_KEYS, update_strategic_ppo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--update-high", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserving existing smoke report")
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload["identity"]["command_schema"] != COMMAND_SCHEMA:
        raise ValueError("Command schema mismatch")
    sample = Path(next(iter(payload["identity"]["validation"])))
    scenario = str(load_command_episode(sample)["scenario_hash"])
    executor = CommandExecutor().to(args.device)
    executor.load_state_dict(payload["actor"])
    controllers = {
        side: HierarchicalController(executor, StrategicActor().to(args.device), seed=1701 + i)
        for i, side in enumerate(("host", "opponent"))
    }
    models = (
        {side: StrategicActorCritic(c.strategy).to(args.device) for side, c in controllers.items()}
        if args.update_high
        else {}
    )
    samples = {side: [] for side in controllers}
    frozen_executor = {k: v.detach().cpu().clone() for k, v in executor.state_dict().items()}
    env = SynchronousExtractionEnv(
        config_path=Path(
            "assets/scenarios/crystal_run_extraction_randomized/crystal_run_extraction_randomized.cfg"
        ),
        seed=0,
        layout_variant=0,
    )
    timeline = []
    previous_health = {side: 100 for side in controllers}
    rewards = {side: [] for side in controllers}
    local_terminal = {side: False for side in controllers}
    maximum_replay_error = 0.0
    try:
        observations, info = env.reset()
        if info.scenario_hash != scenario:
            raise ValueError("Scenario mismatch")
        for decision in range(700):
            condition = (
                ControlCondition()
                if decision < 16
                else ControlCondition(aggressive=1, difficulty=0.5)
            )
            actions = {}
            active = {}
            for side, controller in controllers.items():
                obs = getattr(observations, side)
                if obs.health <= 0 or obs.banked_value > 0:
                    actions[side] = int(MacroAction.IDLE)
                    continue
                record = controller.step(
                    torch.as_tensor(obs.frame.copy(), device=args.device)[None, None, None],
                    torch.as_tensor(extraction_scalars(obs), device=args.device)[None, None],
                    torch.tensor([[obs.previous_action]], device=args.device),
                    condition,
                    public_event=obs.health < previous_health[side],
                )
                previous_health[side] = obs.health
                active[side] = obs.banked_value
                if record["replanned"]:
                    with torch.no_grad():
                        replay = controller.strategy(**controller.last_high_inputs)
                    replay_log_prob = float(replay.logits.log_softmax(-1)[0, 0, record["command"]])
                    maximum_replay_error = max(
                        maximum_replay_error, abs(replay_log_prob - record["command_log_prob"])
                    )
                    if args.update_high:
                        sample = dict(controller.last_high_inputs)
                        sample["privileged"] = extraction_privileged_tensor(
                            env.privileged_state(), learner_side=side, device=args.device
                        )
                        with torch.no_grad():
                            sample["old_values"] = models[side](**sample).values
                        samples[side].append(sample)
                actions[side] = record["action"]
                timeline.append({"side": side, **record})
            observations = env.step(actions["host"], actions["opponent"])
            for side, previous_banked in active.items():
                own = getattr(observations, side)
                rewards[side].append((own.banked_value - previous_banked) / 150)
                local_terminal[side] = (
                    own.health <= 0 or own.banked_value > 0 or observations.terminated
                )
            if observations.terminated or observations.truncated:
                final = env.privileged_state()  # Scoring only, after global settlement.
                transitions = {}
                for side in controllers:
                    rows = [row for row in timeline if row["side"] == side]
                    assert abs(sum(rewards[side]) - getattr(final, f"{side}_banked") / 150) < 1e-8
                    transitions[side] = [
                        asdict(t)
                        for t in aggregate_strategic_transitions(
                            rows,
                            rewards[side],
                            gamma=0.997,
                            terminated=local_terminal[side],
                            truncated=not local_terminal[side],
                        )
                    ]
                if maximum_replay_error > 1e-6:
                    raise RuntimeError("High-level sampled log probability failed replay")
                updates = {}
                if args.update_high:
                    if not all(local_terminal.values()):
                        raise RuntimeError(
                            "Terminal-only smoke needs true local terminals; "
                            "no fabricated truncation bootstrap"
                        )
                    for side, model in models.items():
                        segments = transitions[side]
                        assert len(samples[side]) == len(segments)
                        batch = {
                            k: torch.cat([s[k] for s in samples[side]], 1)
                            for k in (*HIGH_INPUT_KEYS, "privileged", "old_values")
                        }
                        for key, field, dtype in (
                            ("commands", "command", torch.long),
                            ("old_log_probs", "log_prob", torch.float32),
                            ("rewards", "reward", torch.float32),
                            ("durations", "duration", torch.long),
                            ("terminated", "terminated", torch.bool),
                            ("truncated", "truncated", torch.bool),
                        ):
                            batch[key] = torch.tensor(
                                [[t[field] for t in segments]], dtype=dtype, device=args.device
                            )
                        batch["next_values"] = torch.cat(
                            (batch["old_values"][:, 1:], torch.zeros(1, 1, device=args.device)), 1
                        )
                        updates[side] = update_strategic_ppo(
                            model, torch.optim.Adam(model.parameters(), lr=1e-4), batch
                        )
                    if any(
                        not torch.equal(v.detach().cpu(), frozen_executor[k])
                        for k, v in executor.state_dict().items()
                    ):
                        raise RuntimeError("Strategic update modified shared executor")
                report = {
                    "complete": True,
                    "untrained_high_actor": True,
                    "scope": "runtime plumbing only; no learned strategy or control claim",
                    "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    "scenario_hash": scenario,
                    "seed": 0,
                    "torch_seed": 1701,
                    "decisions": decision + 1,
                    "payoffs": own_payoffs(
                        final.host_banked, final.opponent_banked, globally_settled=True
                    ),
                    "timeline": timeline,
                    "strategic_transitions": transitions,
                    "maximum_replay_error": maximum_replay_error,
                    "high_updates": updates,
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                temp = args.output.with_suffix(".tmp")
                temp.write_text(json.dumps(report, indent=2))
                temp.replace(args.output)
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in report.items()
                            if k not in {"timeline", "strategic_transitions"}
                        }
                    ),
                    flush=True,
                )
                return
        raise RuntimeError("Missing global terminal flag")
    finally:
        env.close()


if __name__ == "__main__":
    main()
