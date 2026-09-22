"""Single-command search trials from reset, without post-success waiting."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from botcolosseo.agents.extraction_teachers import PrivilegedStrongExtractionTeacher
from botcolosseo.agents.hierarchical_model import CommandExecutor
from botcolosseo.agents.hierarchical_teacher import PrivilegedCommandTeacher
from botcolosseo.data.extraction_demonstrations import extraction_scalars
from botcolosseo.data.hierarchical_demonstrations import load_command_episode
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv
from botcolosseo.training.hierarchical_protocol import COMMAND_SCHEMA, Command


@torch.no_grad()
def trial(
    actor,
    *,
    seed,
    side,
    command,
    scenario,
    device,
    oracle=False,
    corrections=None,
    teacher_demonstrations=False,
    difficulty=1.0,
):
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError("Difficulty must be finite and within [0,1]")
    if teacher_demonstrations and not oracle:
        raise ValueError("Teacher demonstrations require explicit oracle mode")
    if corrections is not None and (
        (oracle and not teacher_demonstrations) or corrections.exists()
    ):
        raise ValueError("Corrections require fair actor and a fresh output path")
    rows = []

    def finish(result):
        own = getattr(observations, side)
        result.update(
            final_health=float(own.health),
            final_banked=float(own.banked_value),
            terminated=bool(getattr(observations, "terminated", False)),
            truncated=bool(getattr(observations, "truncated", False)),
            difficulty=difficulty,
        )
        if corrections is not None and rows:
            count = len(rows)
            arrays = {
                "frames": np.stack([r[0] for r in rows]),
                "scalars": np.stack([r[1] for r in rows]),
                "previous_actions": np.asarray([r[2] for r in rows], dtype=np.int64),
                "actions": np.asarray([r[3] for r in rows], dtype=np.int64),
                "commands": np.full(count, int(command), dtype=np.int64),
                "valid": np.ones(count, dtype=np.bool_),
                "episode_start": np.arange(count) == 0,
                "command_start": np.arange(count) == 0,
                "difficulty": np.full(count, difficulty, dtype=np.float32),
                "seed": np.asarray(seed),
                "side": np.asarray(side),
                "scenario_hash": np.asarray(scenario),
                "command_schema": np.asarray(COMMAND_SCHEMA),
                "collection_mode": np.asarray(
                    "crossed-search-teacher"
                    if teacher_demonstrations
                    else "single-command-actor-corrections"
                ),
            }
            corrections.parent.mkdir(parents=True, exist_ok=True)
            temp = corrections.with_suffix(".tmp")
            with temp.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            temp.replace(corrections)
            result["valid_labels"] = count
        return result

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
    result = dict(
        seed=seed,
        side=side,
        command=command.name,
        applicable=False,
        success=False,
        decisions=0,
        outcome="unavailable",
    )
    hidden = None
    try:
        observations, info = env.reset()
        initial_obs = getattr(observations, side)
        result["initial_frame_sha256"] = hashlib.sha256(initial_obs.frame.tobytes()).hexdigest()
        result["initial_scalars"] = extraction_scalars(initial_obs).tolist()
        if info.scenario_hash != scenario:
            raise ValueError("Scenario mismatch")
        scorer.begin(command, env.privileged_state())
        result["applicable"] = scorer.act(env.privileged_state()).status == "executing"
        if not result["applicable"]:
            return finish(result)
        for t in range(120):
            obs = getattr(observations, side)
            if corrections is not None:
                label = scorer.act(env.privileged_state())
                if label.status != "executing":
                    raise RuntimeError("Correction requested outside an executable command")
                rows.append(
                    (obs.frame, extraction_scalars(obs), obs.previous_action, int(label.action))
                )
            if oracle:
                action = int(scorer.act(env.privileged_state()).action)
            else:
                output = actor(
                    torch.as_tensor(obs.frame.copy(), device=device)[None, None, None],
                    torch.as_tensor(extraction_scalars(obs), device=device)[None, None],
                    torch.tensor([[obs.previous_action]], device=device),
                    torch.tensor([[float(t > 0)]], device=device),
                    torch.tensor([[int(command)]], device=device),
                    torch.full((1, 1, 1), difficulty, device=device),
                    hidden,
                )
                hidden = output.hidden
                action = int(output.logits.argmax(-1).item())
            actions = {side: action, other: opponent.act(env.privileged_state())}
            before = env.protocol_snapshot()
            observations = env.step(actions["host"], actions["opponent"])
            scorer.observe_transition(before, env.protocol_snapshot())
            status = scorer.act(env.privileged_state()).status
            result.update(decisions=t + 1, success=status == "complete", outcome=status)
            if status != "executing" or observations.terminated or observations.truncated:
                return finish(result)
        result["outcome"] = "budget_exhausted"
        return finish(result)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[96, 97, 98, 99])
    destinations = parser.add_mutually_exclusive_group()
    destinations.add_argument("--corrections-dir", type=Path)
    destinations.add_argument("--teacher-demonstrations-dir", type=Path)
    args = parser.parse_args()
    if not 0.0 <= args.difficulty <= 1.0:
        raise ValueError("Difficulty must be finite and within [0,1]")
    if args.output.exists():
        raise FileExistsError("Preserving existing search probe")
    torch.set_num_threads(2)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    destination = args.corrections_dir or args.teacher_demonstrations_dir
    if destination is not None:
        layouts = {
            int(load_command_episode(Path(p))["seed"]) % 128 for p in payload["identity"]["train"]
        }
        if (args.corrections_dir is not None and args.oracle) or any(
            s % 128 not in layouts for s in args.seeds
        ):
            raise ValueError("Corrections restricted to fair-actor training layouts")
        if args.teacher_demonstrations_dir is not None and not args.oracle:
            raise ValueError("Teacher data requires --oracle")
    if payload["identity"]["command_schema"] != COMMAND_SCHEMA:
        raise ValueError("Command schema mismatch")
    scenario = str(
        load_command_episode(Path(next(iter(payload["identity"]["validation"]))))["scenario_hash"]
    )
    actor = CommandExecutor().to(args.device).eval()
    actor.load_state_dict(payload["actor"])
    report = dict(
        complete=False,
        privileged_oracle=args.oracle,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        scenario_hash=scenario,
        protocol="reset-single-search120-v1",
        difficulty=args.difficulty,
        cases=[],
        scope="skill trial, not full-episode task or payoff evidence",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for command in (Command.SEARCH_NORTH, Command.SEARCH_CENTER, Command.SEARCH_SOUTH):
        for seed in args.seeds:
            for side in ("host", "opponent"):
                row = trial(
                    actor,
                    seed=seed,
                    side=side,
                    command=command,
                    scenario=scenario,
                    device=args.device,
                    oracle=args.oracle,
                    difficulty=args.difficulty,
                    corrections=None
                    if destination is None
                    else destination / f"{command.name}-{seed}-{side}.npz",
                    teacher_demonstrations=args.teacher_demonstrations_dir is not None,
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
