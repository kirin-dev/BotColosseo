"""Paired candidate versus meta-policy rollouts under each condition's own game."""

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
from botcolosseo.training.hierarchical_conditional_population import conditional_marginals
from botcolosseo.training.hierarchical_protocol import matrix_condition


def draw_pair(seed, repeat, role, condition_index, own, other):
    """Sample both identities once; candidate and baseline share the other identity."""
    rng = np.random.default_rng(
        np.random.SeedSequence([1701, seed, repeat, int(role == "opponent"), condition_index])
    )
    return int(rng.choice(len(own), p=own)), int(rng.choice(len(other), p=other))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executor", "candidate", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--condition-solutions", type=Path, nargs="+", required=True)
    parser.add_argument("--population", type=Path, nargs="+", required=True)
    parser.add_argument("--role", choices=("host", "opponent"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--exclude-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    layouts = {s % 128 for s in args.seeds}
    if args.repeats <= 0 or len(layouts) != len(args.seeds) or any(s < 0 for s in args.seeds):
        raise ValueError("Positive repeats and distinct nonnegative layout seeds required")
    solutions = [json.loads(p.read_text()) for p in args.condition_solutions]
    conditions = [matrix_condition(s["identity"]) for s in solutions]
    populations = role_paths(solutions[0]["identity"], args.population)
    for solution in solutions:
        if role_paths(solution["identity"], args.population) != populations:
            raise ValueError("Role populations must agree across conditions")
    executor_hash = digest(args.executor)
    marginals = {
        role: conditional_marginals(
            solutions,
            conditions,
            executor=executor_hash,
            population=[digest(p) for p in populations[role]],
            learner_role="opponent" if role == "host" else "host",
        )
        for role in ("host", "opponent")
    }
    excluded = {s % 128 for s in args.exclude_seeds}
    excluded.update(s % 128 for sol in solutions for s in sol["identity"]["seeds"])
    payload = torch.load(args.executor, map_location="cpu", weights_only=False)
    excluded.update(
        int(load_command_episode(Path(p))["seed"]) % 128 for p in payload["identity"]["train"]
    )
    if layouts & excluded:
        raise ValueError("Evaluation overlaps training, matrix or excluded layouts")
    scenario = str(
        load_command_episode(Path(next(iter(payload["identity"]["validation"]))))["scenario_hash"]
    )
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    executor = CommandExecutor().to(args.device).eval()
    executor.load_state_dict(payload["actor"])
    executor.requires_grad_(False)

    def load_actor(path):
        item = torch.load(path, map_location="cpu", weights_only=False)
        if item["identity"]["executor"] != executor_hash:
            raise ValueError("Actor executor mismatch")
        actor = StrategicActor().to(args.device).eval()
        actor.load_state_dict(item["actor"])
        return actor.requires_grad_(False)

    candidate = load_actor(args.candidate)
    actors = {role: [load_actor(p) for p in paths] for role, paths in populations.items()}
    identity = {
        "executor": executor_hash,
        "candidate": digest(args.candidate),
        "solutions": [digest(p) for p in args.condition_solutions],
        "conditions": [list(c.as_tuple()) for c in conditions],
        "population": {role: [digest(p) for p in paths] for role, paths in populations.items()},
        "marginals": {role: values.tolist() for role, values in marginals.items()},
        "role": args.role,
        "seeds": args.seeds,
        "repeats": args.repeats,
        "excluded_layouts": sorted(excluded),
        "scenario": scenario,
        "protocol": "conditional-own-bank150-candidate-vs-meta-common-opponent-v1",
        "sources": {
            path.name: digest(path)
            for path in (
                Path(__file__),
                Path(__file__).parents[1] / "training/hierarchical_collection.py",
                Path(__file__).parents[1] / "agents/hierarchical_controller.py",
            )
        },
    }
    result = {"identity": identity, "complete": False, "cases": []}
    if args.output.exists():
        if not args.resume:
            raise FileExistsError("Preserve existing response evaluation")
        result = json.loads(args.output.read_text())
        if result["identity"] != identity:
            raise ValueError("Response evaluation resume identity mismatch")
    done = {(r["condition_index"], r["seed"], r["repeat"], r["method"]) for r in result["cases"]}
    if len(done) != len(result["cases"]):
        raise ValueError("Duplicate response evaluation cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(args.output)

    other = "opponent" if args.role == "host" else "host"
    for index, condition in enumerate(conditions):
        for seed in args.seeds:
            for repeat in range(args.repeats):
                own_index, opponent_index = draw_pair(
                    seed,
                    repeat,
                    args.role,
                    index,
                    marginals[args.role][index],
                    marginals[other][index],
                )
                for method in ("baseline", "candidate"):
                    if (index, seed, repeat, method) in done:
                        continue
                    chosen = {
                        args.role: candidate
                        if method == "candidate"
                        else actors[args.role][own_index],
                        other: actors[other][opponent_index],
                    }
                    controllers = {
                        side: HierarchicalController(
                            executor,
                            chosen[side],
                            seed=1701 + 100 * seed + 2 * repeat + side_index,
                        )
                        for side_index, side in enumerate(("host", "opponent"))
                    }
                    env = SynchronousExtractionEnv(
                        config_path=Path(
                            "assets/scenarios/crystal_run_extraction_randomized/"
                            "crystal_run_extraction_randomized.cfg"
                        ),
                        seed=seed,
                        layout_variant=seed % 128,
                    )
                    try:
                        _, report = collect_strategic_episode(
                            env,
                            controllers,
                            StrategicActorCritic(chosen[args.role]).to(args.device),
                            learner_side=args.role,
                            expected_scenario=scenario,
                            condition=condition,
                        )
                    finally:
                        env.close()
                    observed = {
                        (*r["style"], r["difficulty"], r["high_difficulty"])
                        for r in report["timeline"]
                    }
                    if observed != {(*condition.as_tuple(), condition.difficulty)}:
                        raise RuntimeError("Applied evaluation condition mismatch")
                    row = {
                        "condition_index": index,
                        "condition": list(condition.as_tuple()),
                        "seed": seed,
                        "layout": seed % 128,
                        "repeat": repeat,
                        "method": method,
                        "baseline_index": own_index,
                        "opponent_index": opponent_index,
                        "payoff": report["payoffs"][int(args.role == "opponent")],
                        "payoffs": report["payoffs"],
                        "decisions": report["learner_decisions"],
                    }
                    result["cases"].append(row)
                    save()
                    print(json.dumps(row), flush=True)
    result["complete"] = True
    save()


if __name__ == "__main__":
    main()
