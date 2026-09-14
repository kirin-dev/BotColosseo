"""Fair fixed-schedule closed-loop diagnostic; not a learned high-level policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from botcolosseo.agents.extraction_teachers import (
    PrivilegedStrongExtractionTeacher,
    opponent_position,
    player_pose,
)
from botcolosseo.agents.hierarchical_model import CommandExecutor
from botcolosseo.agents.hierarchical_teacher import (
    PrivilegedCommandTeacher,
    counterfactual_command_labels,
)
from botcolosseo.data.extraction_demonstrations import extraction_scalars
from botcolosseo.data.hierarchical_demonstrations import load_command_episode
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv
from botcolosseo.training.extraction_checkpoint import load_extraction_strong_actor
from botcolosseo.training.hierarchical_protocol import COMMAND_SCHEMA, Command


def scheduled_command(
    decision: int, seed: int, profile: str = "search", extraction_endpoint: str | None = None
) -> Command:
    # Fixed schedule depends only on public clock and predeclared case seed.
    if extraction_endpoint not in (None, "north", "south"):
        raise ValueError("Invalid extraction endpoint")
    if profile == "combat":
        if decision < 48:
            return Command.ENGAGE
        if decision < 128:
            return Command.DISENGAGE
    elif profile != "search":
        raise ValueError("Unknown command schedule")
    if decision < 240:
        return Command((decision // 80 + seed % 3) % 3)
    north = seed % 2 == 0 if extraction_endpoint is None else extraction_endpoint == "north"
    return Command.EXTRACT_NORTH if north else Command.EXTRACT_SOUTH


@torch.no_grad()
def evaluate_case(
    actor: CommandExecutor,
    seed: int,
    side: str,
    device: str,
    *,
    expected_scenario: str,
    profile: str = "search",
    corrections: Path | None = None,
    privileged_oracle: bool = False,
    extraction_endpoint: str | None = None,
    oracle_extraction_only: bool = False,
    difficulty: float = 1.0,
) -> dict:
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError("Difficulty must be finite and within [0,1]")
    if oracle_extraction_only and (privileged_oracle or corrections is not None):
        raise ValueError("Extraction oracle cannot mix with full oracle or actor corrections")
    if corrections is not None and corrections.exists():
        raise FileExistsError("Preserving existing correction trajectory")
    env = SynchronousExtractionEnv(
        config_path=Path(
            "assets/scenarios/crystal_run_extraction_randomized/crystal_run_extraction_randomized.cfg"
        ),
        seed=seed,
        layout_variant=seed % 128,
    )
    other = "opponent" if side == "host" else "host"
    opponent = PrivilegedStrongExtractionTeacher(side=other, layout_variant=seed % 128)
    scorer = PrivilegedCommandTeacher(side=side, layout_variant=seed % 128)
    hidden = None
    records = []
    extraction_event = None
    rows = []
    previous_command = None
    actor.eval()
    try:
        observations, info = env.reset()
        if info.scenario_hash != expected_scenario:
            raise ValueError("Evaluation scenario differs from checkpoint data")
        for decision in range(700):
            command = scheduled_command(decision, seed, profile, extraction_endpoint)
            if command != previous_command:
                scorer.begin(command, env.privileged_state())
                initial_status = scorer.act(env.privileged_state()).status
                records.append(
                    {
                        "command": command.name,
                        "start": decision,
                        "completed_at": None,
                        "initial_status": initial_status,
                        "applicable": initial_status == "executing",
                        "initial_health": getattr(observations, side).health,
                        "initial_enemy_distance": math.dist(
                            player_pose(env.privileged_state(), side)[:2],
                            opponent_position(env.privileged_state(), side),
                        ),
                    }
                )
                previous_command = command
            obs = getattr(observations, side)
            out = actor(
                torch.as_tensor(obs.frame.copy(), device=device)[None, None, None],
                torch.as_tensor(extraction_scalars(obs), device=device)[None, None],
                torch.tensor([[obs.previous_action]], device=device),
                torch.tensor([[float(decision > 0)]], device=device),
                torch.tensor([[int(command)]], device=device),
                torch.full((1, 1, 1), difficulty, device=device),
                hidden,
            )
            hidden = out.hidden
            if corrections is not None:
                label = scorer.act(env.privileged_state())
                alternative_labels, alternative_valid = counterfactual_command_labels(
                    env.privileged_state(), side=side, layout_variant=seed % 128
                )
                rows.append(
                    (
                        obs.frame,
                        extraction_scalars(obs),
                        obs.previous_action,
                        int(label.action),
                        int(command),
                        label.status == "executing",
                        decision == 0,
                        records[-1]["start"] == decision,
                        alternative_labels,
                        alternative_valid,
                    )
                )
            actions = {
                side: int(scorer.act(env.privileged_state()).action)
                if privileged_oracle or (oracle_extraction_only and decision >= 240)
                else int(out.logits.argmax(-1)),
                other: opponent.act(env.privileged_state()),
            }
            before = env.protocol_snapshot()
            step = env.step(actions["host"], actions["opponent"])
            if extraction_event is None and getattr(step, side).banked_value > obs.banked_value:
                extraction_event = {
                    "decision": decision + 1,
                    "command": command.name,
                    "pose": list(player_pose(env.privileged_state(), side)),
                    "banked": getattr(step, side).banked_value,
                }
            scorer.observe_transition(before, env.protocol_snapshot())
            if (
                scorer.act(env.privileged_state()).status == "complete"
                and records[-1]["applicable"]
                and records[-1]["completed_at"] is None
            ):
                records[-1]["completed_at"] = decision + 1
            observations = step
            if step.terminated or step.truncated:
                final = env.privileged_state()
                if corrections is not None:
                    corrections.parent.mkdir(parents=True, exist_ok=True)
                    arrays = {
                        key: np.asarray([r[index] for r in rows], dtype=dtype)
                        for index, key, dtype in (
                            (0, "frames", np.uint8),
                            (1, "scalars", np.float32),
                            (2, "previous_actions", np.int64),
                            (3, "actions", np.int64),
                            (4, "commands", np.int64),
                            (5, "valid", np.bool_),
                            (6, "episode_start", np.bool_),
                            (7, "command_start", np.bool_),
                            (8, "counterfactual_actions", np.int64),
                            (9, "counterfactual_valid", np.bool_),
                        )
                    }
                    arrays.update(
                        difficulty=np.full(len(rows), difficulty, dtype=np.float32),
                        seed=np.asarray(seed),
                        side=np.asarray(side),
                        scenario_hash=np.asarray(info.scenario_hash),
                        command_schema=np.asarray(COMMAND_SCHEMA),
                        collection_mode=np.asarray("actor-rollout-teacher-labels"),
                        extraction_endpoint=np.asarray(extraction_endpoint or "seed-default"),
                    )
                    temporary = corrections.with_suffix(".tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(handle, **arrays)
                    temporary.replace(corrections)
                return {
                    "seed": seed,
                    "side": side,
                    "difficulty": difficulty,
                    "scenario_hash": info.scenario_hash,
                    "commands": records,
                    "banked": getattr(final, f"{side}_banked"),
                    "extraction_event": extraction_event,
                    "terminated": step.terminated,
                    "truncated": step.truncated,
                    "decisions": decision + 1,
                }
        raise RuntimeError("Missing terminal flag")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialization-baseline", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[96, 97])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--profile", choices=("search", "combat"), default="search")
    parser.add_argument("--corrections-dir", type=Path)
    parser.add_argument("--privileged-oracle", action="store_true")
    parser.add_argument("--extraction-endpoint", choices=("north", "south"))
    parser.add_argument("--oracle-extraction-only", action="store_true")
    args = parser.parse_args()
    if args.oracle_extraction_only and (
        args.privileged_oracle or args.corrections_dir is not None or args.initialization_baseline
    ):
        raise ValueError("Extraction oracle requires a fair checkpoint prefix and no corrections")
    if args.privileged_oracle and (
        args.corrections_dir is not None or args.initialization_baseline
    ):
        raise ValueError("Oracle diagnostic cannot collect actor corrections or use actor baseline")
    if not 0.0 <= args.difficulty <= 1.0:
        raise ValueError("Difficulty must be finite and within [0,1]")
    if args.output.exists():
        raise FileExistsError("Preserving existing evaluation")
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.corrections_dir is not None:
        training_layouts = {
            int(load_command_episode(Path(p))["seed"]) % 128 for p in payload["identity"]["train"]
        }
        if any(seed % 128 not in training_layouts for seed in args.seeds):
            raise ValueError("Correction labels are restricted to training layouts")
    if payload["identity"]["command_schema"] != COMMAND_SCHEMA:
        raise ValueError("Checkpoint command schema mismatch")
    actor = CommandExecutor()
    actor.load_state_dict(payload["actor"])
    sample = Path(next(iter(payload["identity"]["validation"])))
    scenario = str(load_command_episode(sample)["scenario_hash"])
    if args.initialization_baseline:
        # Resolve the exact original Strong from the checkpoint's verified hash.
        strong = Path("runs/extraction-randomized/strong-ppo-conservative-v2/candidate-0950000.pt")
        if hashlib.sha256(strong.read_bytes()).hexdigest() != payload["identity"]["strong"]:
            raise ValueError("Initialization baseline hash mismatch")
        base, _ = load_extraction_strong_actor(strong, expected_scenario_hash=scenario)
        actor = CommandExecutor()
        actor.initialize_from(base)
    actor.to(args.device)
    result = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "initialization_baseline": args.initialization_baseline,
        "privileged_oracle": args.privileged_oracle,
        "oracle_extraction_only": args.oracle_extraction_only,
        "oracle_start_decision": 240 if args.oracle_extraction_only else None,
        "command_schema": COMMAND_SCHEMA,
        "protocol": "fixed-search240-then-extract",
        "scoring_version": "applicable-region-pickup-endpoint-extraction-v2",
        "profile": args.profile,
        "difficulty": args.difficulty,
        "extraction_endpoint": args.extraction_endpoint,
        "complete": False,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for side in ("host", "opponent"):
            case = evaluate_case(
                actor,
                seed,
                side,
                args.device,
                expected_scenario=scenario,
                profile=args.profile,
                difficulty=args.difficulty,
                extraction_endpoint=args.extraction_endpoint,
                privileged_oracle=args.privileged_oracle,
                oracle_extraction_only=args.oracle_extraction_only,
                corrections=None
                if args.corrections_dir is None
                else (args.corrections_dir / f"{args.profile}-{seed}-{side}.npz"),
            )
            result["cases"].append(case)
            temp = args.output.with_suffix(".tmp")
            temp.write_text(json.dumps(result, indent=2))
            temp.replace(args.output)
            print(json.dumps(case), flush=True)
    result["complete"] = True
    temp = args.output.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(args.output)


if __name__ == "__main__":
    main()
