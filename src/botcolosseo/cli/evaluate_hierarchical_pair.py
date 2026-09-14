"""Paired strategic cross-play with independent own-bank payoffs."""

import argparse
import json
from pathlib import Path

import torch

from botcolosseo.agents.hierarchical_controller import HierarchicalController
from botcolosseo.agents.hierarchical_critic import StrategicActorCritic
from botcolosseo.agents.hierarchical_model import CommandExecutor, StrategicActor
from botcolosseo.cli.train_hierarchical_strategic import digest
from botcolosseo.data.hierarchical_demonstrations import load_command_episode
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv
from botcolosseo.training.hierarchical_collection import collect_strategic_episode
from botcolosseo.training.hierarchical_protocol import ControlCondition


def main():
    parser = argparse.ArgumentParser()
    for name in ("executor", "first", "second", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(96, 108)))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--condition", type=float, nargs=4, default=[0, 0, 0, 1],
                        metavar=("A", "D", "E", "DIFFICULTY"))
    args = parser.parse_args()
    condition = ControlCondition(*args.condition)
    if args.repeats <= 0 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Positive repeats and unique seeds required")
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    executor_payload = torch.load(args.executor, map_location="cpu", weights_only=False)
    scenario = str(load_command_episode(
        Path(next(iter(executor_payload["identity"]["validation"])))
    )["scenario_hash"])
    executor = CommandExecutor().to(args.device)
    executor.load_state_dict(executor_payload["actor"])
    executor.requires_grad_(False)
    actors = []
    for path in (args.first, args.second):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["identity"]["executor"] != digest(args.executor):
            raise ValueError("Cross-play requires one executor version")
        actor = StrategicActor().to(args.device)
        actor.load_state_dict(payload["actor"])
        actor.requires_grad_(False)
        actors.append(actor)
    identity = {
        "executor": digest(args.executor), "first": digest(args.first),
        "second": digest(args.second), "scenario": scenario,
        "seeds": args.seeds, "repeats": args.repeats,
        "protocol": "neutral-hard-high-sample-low-argmax-own-bank150-v1",
        "sources": {
            str(path.name): digest(path)
            for path in (
                Path(__file__),
                Path(__file__).parents[1] / "training/hierarchical_collection.py",
                Path(__file__).parents[1] / "agents/hierarchical_controller.py",
            )
        },
    }
    identity["condition"] = list(condition.as_tuple())
    if condition != ControlCondition():
        identity["protocol"] = "fixed-condition-high-sample-low-argmax-own-bank150-v1"
    result = {"identity": identity, "complete": False, "cases": []}
    if args.output.exists():
        if not args.resume:
            raise FileExistsError("Preserve existing cross-play; use --resume")
        result = json.loads(args.output.read_text())
        if result["identity"] != identity:
            raise ValueError("Cross-play resume identity mismatch")
    done = {(row["seed"], row["repeat"], row["first_side"]) for row in result["cases"]}
    if len(done) != len(result["cases"]):
        raise ValueError("Duplicate cross-play cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for repeat in range(args.repeats):
            for first_side in ("host", "opponent"):
                if (seed, repeat, first_side) in done:
                    continue
                controllers = {
                    side: HierarchicalController(
                        executor, actors[0 if side == first_side else 1],
                        seed=1701 + 100 * seed + 2 * repeat + index,
                    ) for index, side in enumerate(("host", "opponent"))
                }
                model = StrategicActorCritic(controllers[first_side].strategy).to(args.device)
                env = SynchronousExtractionEnv(
                    config_path=Path("assets/scenarios/crystal_run_extraction_randomized/crystal_run_extraction_randomized.cfg"),
                    seed=seed, layout_variant=seed % 128,
                )
                try:
                    _, report = collect_strategic_episode(
                        env, controllers, model, learner_side=first_side,
                        expected_scenario=scenario, condition=condition
                    )
                finally:
                    env.close()
                payoffs = report["payoffs"]
                observed = {
                    (*step["style"], step["difficulty"], step["high_difficulty"])
                    for step in report["timeline"]
                }
                expected = (*condition.as_tuple(), condition.difficulty)
                if observed != {expected}:
                    raise RuntimeError("Applied rollout condition differs from the matrix")
                row = {
                    "seed": seed, "layout": seed % 128, "repeat": repeat,
                    "first_side": first_side, "host_payoff": payoffs[0],
                    "opponent_payoff": payoffs[1],
                    "first_payoff": payoffs[0 if first_side == "host" else 1],
                    "second_payoff": payoffs[1 if first_side == "host" else 0],
                    "global_decisions": report["global_decisions"],
                    "applied_condition": list(condition.as_tuple()),
                }
                result["cases"].append(row)
                temporary = args.output.with_suffix(".tmp")
                temporary.write_text(json.dumps(result, indent=2))
                temporary.replace(args.output)
                print(json.dumps(row), flush=True)
    result["complete"] = True
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
