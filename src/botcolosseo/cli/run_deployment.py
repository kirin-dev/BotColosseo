"""Run the portable inference bundle without training data or original paths."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from botcolosseo.agents.hierarchical_controller import HierarchicalController
from botcolosseo.agents.hierarchical_critic import StrategicActorCritic
from botcolosseo.agents.hierarchical_model import CommandExecutor, StrategicActor
from botcolosseo.demo.hierarchical_controls import ScheduledController, control_schedule
from botcolosseo.demo.hierarchical_recording import StrategicRecordingEnv
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv
from botcolosseo.envs.video import write_mp4
from botcolosseo.training.hierarchical_collection import collect_strategic_episode
from botcolosseo.training.hierarchical_protocol import ControlCondition


def load_bundle(root):
    root = root.resolve()
    manifest = json.loads((root / "deployment.json").read_text())
    if manifest["schema"] != "botcolosseo-deployment-1":
        raise ValueError("Unsupported deployment bundle")
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if (not path.is_relative_to(root)
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError(f"Bundle integrity failure: {name}")
    models = []
    for name, cls in (("executor", CommandExecutor), ("strategy", StrategicActor),
                      ("opponent", StrategicActor)):
        state = torch.load(root / "models" / f"{name}.pt", map_location="cpu", weights_only=True)
        model = cls().eval()
        model.load_state_dict(state, strict=True)
        models.append(model)
    return manifest, models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("."))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--seed", type=int, default=104)
    parser.add_argument("--role", choices=("host", "opponent"), default="host")
    parser.add_argument("--style", choices=("neutral", "aggressive", "defensive", "explorer"),
                        default="neutral")
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--switch-to", choices=("aggressive", "defensive", "explorer"))
    parser.add_argument("--output", type=Path, default=Path("demo.mp4"))
    args = parser.parse_args()
    styles = {"neutral": (0, 0, 0), "aggressive": (1, 0, 0),
              "defensive": (0, 1, 0), "explorer": (0, 0, 1)}
    condition = ControlCondition(*styles[args.style], args.difficulty)
    if args.switch_to and args.style != "neutral":
        raise ValueError("Single-switch demonstration must start Neutral")
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    manifest, (low, high, opponent) = load_bundle(args.bundle)
    if args.check_only:
        print("Bundle integrity and strict model loading: PASS")
        return
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("Preserve previous demo outputs")
    schedule = (control_schedule(difficulty=args.difficulty, single_style=args.switch_to)
                if args.switch_to else [(0, condition)])
    controllers = {
        side: (ScheduledController(low, high, seed=1701 + 100 * args.seed + index,
                                   schedule=schedule) if side == args.role else
               HierarchicalController(low, opponent, seed=1701 + 100 * args.seed + index))
        for index, side in enumerate(("host", "opponent"))
    }
    env = SynchronousExtractionEnv(config_path=args.bundle.resolve() / manifest["scenario_config"],
                                  seed=args.seed, layout_variant=args.seed % 128)
    recording = StrategicRecordingEnv(env, controllers[args.role], side=args.role,
                                      label="PORTABLE BOT")
    try:
        _, report = collect_strategic_episode(
            recording, controllers, StrategicActorCritic(high), learner_side=args.role,
            expected_scenario=manifest["scenario_hash"], condition=ControlCondition())
    finally:
        env.close()
    write_mp4((frame for frame in recording.frames for _ in range(4)), args.output, fps=35)
    evidence = {"source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
                "condition": condition.as_tuple(), "switch_to": args.switch_to,
                "seed": args.seed, "role": args.role, "report": report,
                "events": recording.events, "frames": len(recording.frames) * 4, "fps": 35}
    args.output.with_suffix(".json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps({"payoffs": report["payoffs"], "video": str(args.output)}))


if __name__ == "__main__":
    main()
