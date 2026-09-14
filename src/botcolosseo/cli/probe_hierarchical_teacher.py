"""Small live D0 command-chain probe, not fair-policy evaluation."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from botcolosseo.agents.extraction_teachers import (
    PrivilegedStrongExtractionTeacher,
    opponent_position,
    player_pose,
)
from botcolosseo.agents.hierarchical_teacher import PrivilegedCommandTeacher
from botcolosseo.data.extraction_demonstrations import extraction_scalars
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv
from botcolosseo.training.hierarchical_protocol import COMMAND_SCHEMA, Command, own_payoffs


def run_case(
    seed: int,
    side: str,
    config: Path,
    *,
    trajectory: Path | None = None,
    sustained_combat: bool = False,
) -> dict:
    if trajectory is not None and trajectory.exists():
        raise FileExistsError(f"Preserving trajectory: {trajectory}")
    env = SynchronousExtractionEnv(config_path=config, seed=seed, layout_variant=seed % 128)
    teacher = PrivilegedCommandTeacher(side=side, layout_variant=seed % 128)
    other = "opponent" if side == "host" else "host"
    opponent = PrivilegedStrongExtractionTeacher(side=other, layout_variant=seed % 128)
    commands = [
        Command.SEARCH_NORTH,
        Command.SEARCH_CENTER,
        Command.SEARCH_SOUTH,
        Command.EXTRACT_NORTH if seed % 2 == 0 else Command.EXTRACT_SOUTH,
    ]
    if trajectory is not None:
        # Rotate skills rather than encode one fixed whole-episode strategy.
        skills = list(Command)[:3]
        offset = seed % len(skills)
        commands = skills[offset:] + skills[:offset]
        if seed % 2:
            commands = commands[:1] + [Command.ENGAGE, Command.DISENGAGE] + commands[1:]
        commands += [Command.EXTRACT_NORTH if (seed // 2) % 2 == 0 else Command.EXTRACT_SOUTH]
    rows = []
    records = []
    index = 0
    started = time.monotonic()
    try:
        observations, info = env.reset()
        teacher.begin(commands[index], env.privileged_state())
        command_start = 0
        contact_steps = 0
        for decision in range(700):
            state = env.privileged_state()
            label = teacher.act(state)
            if commands[index] == Command.ENGAGE:
                close = (
                    math.dist(player_pose(state, side)[:2], opponent_position(state, side)) <= 480
                )
                contact_steps = contact_steps + 1 if close else 0
            budget_exhausted = decision - command_start >= 120 or (
                trajectory is not None
                and not sustained_combat
                and commands[index] == Command.ENGAGE
                and contact_steps >= 2
            )
            if index < len(commands) - 1 and (
                label.status in {"complete", "unavailable"} or budget_exhausted
            ):
                records.append(
                    {
                        "command": commands[index].name,
                        "status": label.status,
                        "budget_exhausted": budget_exhausted,
                        "decisions": decision - command_start,
                    }
                )
                index += 1
                command_start = decision
                contact_steps = 0
                teacher.begin(commands[index], state)
                label = teacher.act(state)
            actions = {side: label.action, other: opponent.act(state)}
            observation = getattr(observations, side)
            if trajectory is not None:
                rows.append(
                    (
                        observation.frame,
                        extraction_scalars(observation),
                        observation.previous_action,
                        int(label.action),
                        int(commands[index]),
                        label.status == "executing",
                        decision == 0,
                        decision == command_start,
                    )
                )
            before = env.protocol_snapshot()
            step = env.step(actions["host"], actions["opponent"])
            teacher.observe_transition(before, env.protocol_snapshot())
            observations = step
            if step.terminated or step.truncated:
                final = env.privileged_state()
                records.append(
                    {
                        "command": commands[index].name,
                        "status": teacher.act(final).status,
                        "decisions": decision + 1 - command_start,
                    }
                )
                result = {
                    "seed": seed,
                    "side": side,
                    "scenario_hash": info.scenario_hash,
                    "privileged_oracle": True,
                    "command_schema": COMMAND_SCHEMA,
                    "commands": records,
                    "decisions": decision + 1,
                    "terminated": step.terminated,
                    "truncated": step.truncated,
                    "host_banked": final.host_banked,
                    "opponent_banked": final.opponent_banked,
                    "payoffs": own_payoffs(
                        final.host_banked, final.opponent_banked, globally_settled=True
                    )
                    if step.terminated
                    else None,
                    "wall_seconds": time.monotonic() - started,
                }
                if trajectory is not None:
                    trajectory.parent.mkdir(parents=True, exist_ok=True)
                    temporary = trajectory.with_suffix(".tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(
                            handle,
                            frames=np.stack([r[0] for r in rows]),
                            scalars=np.stack([r[1] for r in rows]),
                            previous_actions=np.asarray([r[2] for r in rows], dtype=np.int64),
                            actions=np.asarray([r[3] for r in rows], dtype=np.int64),
                            commands=np.asarray([r[4] for r in rows], dtype=np.int64),
                            valid=np.asarray([r[5] for r in rows], dtype=np.bool_),
                            episode_start=np.asarray([r[6] for r in rows], dtype=np.bool_),
                            command_start=np.asarray([r[7] for r in rows], dtype=np.bool_),
                            difficulty=np.ones(len(rows), dtype=np.float32),
                            command_schema=np.asarray(COMMAND_SCHEMA),
                            seed=np.asarray(seed),
                            side=np.asarray(side),
                            scenario_hash=np.asarray(info.scenario_hash),
                            sustained_combat=np.asarray(sustained_combat),
                        )
                    temporary.replace(trajectory)
                    result["trajectory"] = str(trajectory)
                    result["valid_labels"] = sum(r[5] for r in rows)
                return result
        raise RuntimeError("Probe exceeded environment horizon without terminal flag")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--sustained-combat", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserving existing probe: {args.output}")
    config = Path(
        "assets/scenarios/crystal_run_extraction_randomized/crystal_run_extraction_randomized.cfg"
    )
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for side in ("host", "opponent"):
            trajectory = (
                None
                if args.trajectory_dir is None
                else (args.trajectory_dir / f"seed-{seed}-{side}.npz")
            )
            result = run_case(
                seed,
                side,
                config,
                trajectory=trajectory,
                sustained_combat=args.sustained_combat,
            )
            results.append(result)
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps({"cases": results, "complete": False}, indent=2))
            temporary.replace(args.output)
            print(json.dumps(result), flush=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps({"cases": results, "complete": True}, indent=2))
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
