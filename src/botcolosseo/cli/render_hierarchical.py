"""Record an actual hierarchical pair; no manual command or action overrides."""

import argparse
import json
from pathlib import Path

import torch

from botcolosseo.agents.hierarchical_controller import HierarchicalController
from botcolosseo.agents.hierarchical_critic import StrategicActorCritic
from botcolosseo.agents.hierarchical_model import CommandExecutor, StrategicActor
from botcolosseo.cli.train_hierarchical_strategic import digest
from botcolosseo.data.hierarchical_demonstrations import load_command_episode
from botcolosseo.demo.hierarchical_controls import ScheduledController, control_schedule
from botcolosseo.demo.hierarchical_recording import StrategicRecordingEnv
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv
from botcolosseo.envs.video import write_mp4
from botcolosseo.training.hierarchical_collection import collect_strategic_episode
from botcolosseo.training.hierarchical_protocol import ControlCondition


def main():
    parser = argparse.ArgumentParser()
    for name in ("executor", "strategy", "opponent", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=96)
    parser.add_argument("--role", choices=("host", "opponent"), default="host")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--label", default="RESEARCH PREVIEW")
    parser.add_argument("--style", type=float, nargs=3, default=[0, 0, 0])
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--switch-mode", choices=("style", "difficulty", "joint"), default="style")
    parser.add_argument(
        "--switch", action="store_true", help="Neutral/A/D/E at decisions 0/81/161/241"
    )
    args = parser.parse_args()
    condition = ControlCondition(*args.style, difficulty=args.difficulty)
    if args.switch_mode != "style" and not args.switch:
        raise ValueError("A switch mode requires --switch")
    schedule = [(0, condition)]
    if args.switch:
        schedule = control_schedule(args.switch_mode, difficulty=args.difficulty)
    evidence = args.output.with_suffix(".json")
    if args.output.exists() or evidence.exists():
        raise FileExistsError("Preserving recorded video and evidence")
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    payload = torch.load(args.executor, map_location="cpu", weights_only=False)
    scenario = str(
        load_command_episode(Path(next(iter(payload["identity"]["validation"]))))["scenario_hash"]
    )
    low = CommandExecutor().to(args.device)
    low.load_state_dict(payload["actor"])
    controllers = {}
    for index, side in enumerate(("host", "opponent")):
        path = args.strategy if side == args.role else args.opponent
        high = StrategicActor().to(args.device)
        item = torch.load(path, map_location="cpu", weights_only=False)
        if item["identity"]["executor"] != digest(args.executor):
            raise ValueError("Recording must use one executor version")
        high.load_state_dict(item["actor"])
        controllers[side] = HierarchicalController(low, high, seed=1701 + 100 * args.seed + index)
        if side == args.role:
            controllers[side] = ScheduledController(
                low, high, seed=1701 + 100 * args.seed + index, schedule=schedule
            )
    env = SynchronousExtractionEnv(
        config_path=Path(
            "assets/scenarios/crystal_run_extraction_randomized/crystal_run_extraction_randomized.cfg"
        ),
        seed=args.seed,
        layout_variant=args.seed % 128,
    )
    recording = StrategicRecordingEnv(env, controllers[args.role], side=args.role, label=args.label)
    try:
        _, report = collect_strategic_episode(
            recording,
            controllers,
            StrategicActorCritic(controllers[args.role].strategy).to(args.device),
            learner_side=args.role,
            expected_scenario=scenario,
        )
    finally:
        env.close()
    # Each low action spans four 35Hz engine ticks: no playback-speed inflation.
    write_mp4((frame for frame in recording.frames for _ in range(4)), args.output, fps=35)
    result = {
        "executor": digest(args.executor),
        "strategy": digest(args.strategy),
        "opponent": digest(args.opponent),
        "video": digest(args.output),
        "scenario": scenario,
        "seed": args.seed,
        "learner_role": args.role,
        "layout": args.seed % 128,
        "control_schedule": [
            {"decision": time, "condition": value.as_tuple()} for time, value in schedule
        ],
        "opponent_condition": ControlCondition().as_tuple(),
        "frames": len(recording.frames) * 4,
        "fps": 35,
        "events": recording.events,
        "command_changes": recording.command_changes,
        "report": report,
        "scope": "actual research rollout, not a representative-style claim",
    }
    temporary = evidence.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(evidence)
    print(json.dumps({key: value for key, value in result.items() if key != "report"}), flush=True)


if __name__ == "__main__":
    main()
