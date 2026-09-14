"""Controlled-start disengagement diagnostic, not full-episode policy evidence."""

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
    steer_toward,
)
from botcolosseo.agents.hierarchical_model import CommandExecutor
from botcolosseo.agents.hierarchical_teacher import PrivilegedCommandTeacher
from botcolosseo.data.extraction_demonstrations import extraction_scalars
from botcolosseo.data.hierarchical_demonstrations import load_command_episode
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv
from botcolosseo.training.hierarchical_protocol import COMMAND_SCHEMA, Command


@torch.no_grad()
def trial(
    actor,
    *,
    seed,
    side,
    scenario,
    device,
    oracle,
    lateral=False,
    bounded_lateral=False,
    demonstrations=None,
    corrections=False,
):
    if lateral and not (oracle or corrections):
        raise ValueError("Lateral probe requires explicit oracle mode")
    if bounded_lateral and not lateral:
        raise ValueError("Bounded lateral probe requires lateral mode")
    if demonstrations is not None and (not (oracle or corrections) or demonstrations.exists()):
        raise ValueError("Demonstrations require oracle and fresh destination")
    rows = []

    def finish(result):
        if demonstrations is not None and rows:
            commands = np.asarray([r[4] for r in rows], dtype=np.int64)
            arrays = dict(
                frames=np.stack([r[0] for r in rows]),
                scalars=np.stack([r[1] for r in rows]),
                previous_actions=np.asarray([r[2] for r in rows], dtype=np.int64),
                actions=np.asarray([r[3] for r in rows], dtype=np.int64),
                commands=commands,
                valid=np.asarray([r[5] for r in rows], dtype=np.bool_),
                episode_start=np.arange(len(rows)) == 0,
                command_start=np.r_[True, commands[1:] != commands[:-1]],
                difficulty=np.ones(len(rows), dtype=np.float32),
                seed=np.asarray(seed),
                side=np.asarray(side),
                scenario_hash=np.asarray(scenario),
                command_schema=np.asarray(COMMAND_SCHEMA),
                collection_mode=np.asarray(
                    "controlled-disengage-corrections-v2"
                    if corrections
                    else "controlled-disengage-latched-v2"
                ),
            )
            demonstrations.parent.mkdir(parents=True, exist_ok=True)
            temp = demonstrations.with_suffix(".tmp")
            with temp.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            load_command_episode(temp)
            temp.replace(demonstrations)
            result["valid_labels"] = int(arrays["valid"].sum())
        return result

    env = SynchronousExtractionEnv(
        config_path=Path(
            "assets/scenarios/crystal_run_extraction_randomized/crystal_run_extraction_randomized.cfg"
        ),
        seed=seed,
        layout_variant=seed % 128,
    )
    other = "opponent" if side == "host" else "host"
    enemy = PrivilegedStrongExtractionTeacher(side=other, layout_variant=seed % 128)
    scorer = PrivilegedCommandTeacher(side=side, layout_variant=seed % 128)
    hidden = None
    started = None
    lateral_finished = False
    row = dict(seed=seed, side=side, applicable=False, success=False, outcome="preparation_failed")
    try:
        observations, info = env.reset()
        if info.scenario_hash != scenario:
            raise ValueError("Scenario mismatch")
        for decision in range(240):
            state = env.privileged_state()
            obs = getattr(observations, side)
            distance = math.dist(player_pose(state, side)[:2], opponent_position(state, side))
            if started is None and distance <= 400:
                started = decision
                scorer.begin(Command.DISENGAGE, state)
                row.update(
                    applicable=scorer.act(state).status == "executing",
                    preparation_steps=decision,
                    initial_distance=distance,
                    initial_health=obs.health,
                )
                if not row["applicable"]:
                    return finish(row)
            command = Command.ENGAGE if started is None else Command.DISENGAGE
            output = actor(
                torch.as_tensor(obs.frame.copy(), device=device)[None, None, None],
                torch.as_tensor(extraction_scalars(obs), device=device)[None, None],
                torch.tensor([[obs.previous_action]], device=device),
                torch.tensor([[float(decision > 0)]], device=device),
                torch.tensor([[int(command)]], device=device),
                torch.ones(1, 1, 1, device=device),
                hidden,
            )
            hidden = output.hidden
            if started is None:
                # Standardized privileged setup; actor only observes its actual history.
                actions = {
                    side: steer_toward(state, side, opponent_position(state, side)),
                    other: MacroAction.IDLE,
                }
            else:
                label_action = scorer.act(state).action
                if bounded_lateral and abs(player_pose(state, side)[1]) >= 320:
                    lateral_finished = True
                if lateral and not lateral_finished:
                    label_action = MacroAction.STRAFE_LEFT
                actions = {
                    side: label_action if oracle else int(output.logits.argmax(-1)),
                    other: enemy.act(state),
                }
            if demonstrations is not None:
                rows.append(
                    (
                        obs.frame.copy(),
                        extraction_scalars(obs),
                        obs.previous_action,
                        int(label_action) if started is not None else int(actions[side]),
                        int(command),
                        started is not None,
                    )
                )
            observations = env.step(int(actions["host"]), int(actions["opponent"]))
            if started is not None:
                row.setdefault("trace", []).append(
                    {
                        "distance_before": distance,
                        "self_pose_before": list(player_pose(state, side)),
                        "enemy_position_before": list(opponent_position(state, side)),
                        "health_before": obs.health,
                        "action": int(actions[side]),
                        "health_after": getattr(observations, side).health,
                    }
                )
                status = scorer.act(env.privileged_state()).status
                row.update(
                    outcome=status, success=status == "complete", decisions=decision - started + 1
                )
                if status != "executing":
                    return finish(row)
                if decision - started + 1 >= 120:
                    row["outcome"] = "budget_exhausted"
                    return finish(row)
            if observations.terminated or observations.truncated:
                return finish(row)
        return finish(row)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--lateral", action="store_true")
    parser.add_argument("--bounded-lateral", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[96, 97, 98, 99])
    parser.add_argument("--demonstrations-dir", type=Path)
    parser.add_argument("--corrections", action="store_true")
    args = parser.parse_args()
    if args.corrections and (args.oracle or args.demonstrations_dir is None):
        raise ValueError("Corrections require fair actor and a data destination")
    if args.output.exists():
        raise FileExistsError("Preserving diagnostic")
    torch.set_num_threads(2)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.demonstrations_dir is not None:
        layouts = {
            int(load_command_episode(Path(p))["seed"]) % 128 for p in payload["identity"]["train"]
        }
        if not ((args.oracle or args.corrections) and args.lateral and args.bounded_lateral) or any(
            s % 128 not in layouts for s in args.seeds
        ):
            raise ValueError("Demonstrations require latched oracle and training layouts")
    if payload["identity"]["command_schema"] != COMMAND_SCHEMA:
        raise ValueError("Command schema mismatch")
    scenario = str(
        load_command_episode(Path(next(iter(payload["identity"]["validation"]))))["scenario_hash"]
    )
    actor = CommandExecutor().to(args.device).eval()
    actor.load_state_dict(payload["actor"])
    report = dict(
        complete=False,
        privileged_preparation=True,
        privileged_oracle=args.oracle,
        lateral_probe=args.lateral,
        bounded_lateral_probe=args.bounded_lateral,
        actor_corrections=args.corrections,
        protocol="controlled-contact400-disengage120-latched-v2",
        scenario_hash=scenario,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        cases=[],
        scope="controlled-start skill only; preparation is not fair-policy behavior",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for side in ("host", "opponent"):
            row = trial(
                actor,
                seed=seed,
                side=side,
                scenario=scenario,
                device=args.device,
                oracle=args.oracle,
                lateral=args.lateral,
                bounded_lateral=args.bounded_lateral,
                corrections=args.corrections,
                demonstrations=None
                if args.demonstrations_dir is None
                else args.demonstrations_dir / f"disengage-{seed}-{side}.npz",
            )
            report["cases"].append(row)
            temp = args.output.with_suffix(".tmp")
            temp.write_text(json.dumps(report, indent=2))
            temp.replace(args.output)
            print(json.dumps(row), flush=True)
    report["complete"] = True
    temp = args.output.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2))
    temp.replace(args.output)


if __name__ == "__main__":
    main()
