"""Frozen stochastic-policy training-distribution audit; no optimizer updates."""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import torch

from botcolosseo.agents.extraction_teachers import PrivilegedStrongExtractionTeacher
from botcolosseo.agents.hierarchical_model import CommandActorCritic, CommandExecutor
from botcolosseo.cli.evaluate_hierarchical_executor import scheduled_command
from botcolosseo.data.hierarchical_demonstrations import load_command_episode
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv
from botcolosseo.training.hierarchical_rollout import collect_command_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserving audit")
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    layouts = {
        int(load_command_episode(Path(p))["seed"]) % 128 for p in payload["identity"]["train"]
    }
    if not set(range(4)) <= layouts:
        raise ValueError("Audit layouts must belong to training split")
    scenario = str(
        load_command_episode(Path(next(iter(payload["identity"]["train"]))))["scenario_hash"]
    )
    model = CommandActorCritic(CommandExecutor()).to(args.device).requires_grad_(False).eval()
    model.actor.load_state_dict(payload["actor"])
    report = dict(
        complete=False,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        gamma=0.997,
        inference="stochastic",
        scope="training-distribution diagnostic, no updates",
        cases=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in range(4):
        for side in ("host", "opponent"):
            profile = "search" if seed % 2 == 0 else "combat"
            env = SynchronousExtractionEnv(
                config_path=Path(
                    "assets/scenarios/crystal_run_extraction_randomized/crystal_run_extraction_randomized.cfg"
                ),
                seed=seed,
                layout_variant=seed,
            )
            try:
                _, summary = collect_command_episode(
                    model,
                    env,
                    side=side,
                    layout_variant=seed,
                    schedule=partial(scheduled_command, seed=seed, profile=profile),
                    opponent=PrivilegedStrongExtractionTeacher(
                        side="opponent" if side == "host" else "host", layout_variant=seed
                    ),
                    device=args.device,
                    gamma=0.997,
                )
            finally:
                env.close()
            if summary["scenario_hash"] != scenario:
                raise ValueError("Scenario mismatch")
            report["cases"].append(dict(seed=seed, side=side, profile=profile, **summary))
            report["complete"] = len(report["cases"]) == 8
            temp = args.output.with_suffix(".tmp")
            temp.write_text(json.dumps(report, indent=2))
            temp.replace(args.output)


if __name__ == "__main__":
    main()
