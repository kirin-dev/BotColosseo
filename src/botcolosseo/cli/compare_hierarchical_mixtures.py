"""Independent fixed/uniform/meta deployment comparison against one common mixture."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from botcolosseo.agents.hierarchical_controller import HierarchicalController
from botcolosseo.agents.hierarchical_critic import StrategicActorCritic
from botcolosseo.agents.hierarchical_model import CommandExecutor, StrategicActor
from botcolosseo.cli.train_hierarchical_strategic import digest
from botcolosseo.data.hierarchical_demonstrations import load_command_episode
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv
from botcolosseo.evaluation.hierarchical_role_paths import role_paths
from botcolosseo.training.hierarchical_collection import collect_strategic_episode
from botcolosseo.training.hierarchical_protocol import require_neutral_matrix


def sample_indices(seed, repeat, role, method, sizes, meta):
    """Common opponent draw across methods; sample identity once per episode."""
    other = "opponent" if role == "host" else "host"
    base = 1701 + seed * 100 + repeat * 2 + int(role == "opponent")
    opponent = int(np.random.default_rng(base).integers(sizes[other]))
    if method == "fixed":
        learner = 1 if role == "host" else 0  # Prior seed-window Upgrade / Preserve.
    elif method in ("uniform", "meta"):
        probabilities = np.ones(sizes[role]) / sizes[role] if method == "uniform" else meta[role]
        learner = int(np.random.default_rng(base + 1000000).choice(sizes[role], p=probabilities))
    else:
        raise ValueError("Unknown mixture method")
    return learner, opponent


def main():
    parser = argparse.ArgumentParser()
    for name in ("executor", "solution", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--population", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(116, 128)))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    solution = json.loads(args.solution.read_text())
    require_neutral_matrix(solution["identity"])
    if args.repeats <= 0 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Positive repeats and unique seeds required")
    if {s % 128 for s in args.seeds} & {s % 128 for s in solution["identity"]["seeds"]}:
        raise ValueError("Comparison must not reuse matrix layouts")
    if (
        solution["identity"]["executor"] != digest(args.executor)
        or solution["maximum_regret"] > 1e-3
    ):
        raise ValueError("Invalid solution or executor")
    populations = role_paths(solution["identity"], args.population)
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    payload = torch.load(args.executor, map_location="cpu", weights_only=False)
    scenario = str(
        load_command_episode(Path(next(iter(payload["identity"]["validation"]))))["scenario_hash"]
    )
    training_layouts = {
        int(load_command_episode(Path(p))["seed"]) % 128 for p in payload["identity"]["train"]
    }
    if {s % 128 for s in args.seeds} & training_layouts:
        raise ValueError("Comparison overlaps executor training layouts")
    executor = CommandExecutor().to(args.device)
    executor.load_state_dict(payload["actor"])
    executor.eval().requires_grad_(False)
    actors = {}
    for role, paths in populations.items():
        actors[role] = []
        for path in paths:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved["identity"]["executor"] != digest(args.executor):
                raise ValueError("Population executor mismatch")
            actor = StrategicActor().to(args.device)
            actor.load_state_dict(saved["actor"])
            actors[role].append(actor.eval().requires_grad_(False))
    sizes = {role: len(items) for role, items in actors.items()}
    identity = {
        "executor": digest(args.executor),
        "solution": digest(args.solution),
        "population": {role: [digest(p) for p in paths] for role, paths in populations.items()},
        "seeds": args.seeds,
        "repeats": args.repeats,
        "scenario": scenario,
        "protocol": "neutral-hard-fixed-uniform-meta-vs-common-uniform-v1",
        "source": digest(Path(__file__)),
    }
    result = {"identity": identity, "complete": False, "cases": []}
    if args.output.exists():
        if not args.resume:
            raise FileExistsError("Preserve comparison; use --resume")
        result = json.loads(args.output.read_text())
        if result["identity"] != identity:
            raise ValueError("Resume identity mismatch")
    done = {(r["seed"], r["repeat"], r["role"], r["method"]) for r in result["cases"]}
    if len(done) != len(result["cases"]):
        raise ValueError("Duplicate comparison cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for repeat in range(args.repeats):
            for role in ("host", "opponent"):
                other = "opponent" if role == "host" else "host"
                for method in ("fixed", "uniform", "meta"):
                    if (seed, repeat, role, method) in done:
                        continue
                    learner, opponent = sample_indices(seed, repeat, role, method, sizes, solution)
                    chosen = {role: learner, other: opponent}
                    controllers = {
                        side: HierarchicalController(
                            executor,
                            actors[side][chosen[side]],
                            seed=1701 + 100 * seed + 2 * repeat + index,
                        )
                        for index, side in enumerate(("host", "opponent"))
                    }
                    model = StrategicActorCritic(controllers[role].strategy).to(args.device)
                    env = SynchronousExtractionEnv(
                        config_path=Path(
                            "assets/scenarios/crystal_run_extraction_randomized/crystal_run_extraction_randomized.cfg"
                        ),
                        seed=seed,
                        layout_variant=seed % 128,
                    )
                    try:
                        _, report = collect_strategic_episode(
                            env, controllers, model, learner_side=role, expected_scenario=scenario
                        )
                    finally:
                        env.close()
                    row = {
                        "seed": seed,
                        "layout": seed % 128,
                        "repeat": repeat,
                        "role": role,
                        "method": method,
                        "learner_index": learner,
                        "opponent_index": opponent,
                        "payoff": report["payoffs"][int(role == "opponent")],
                        "payoffs": report["payoffs"],
                        "global_decisions": report["global_decisions"],
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
