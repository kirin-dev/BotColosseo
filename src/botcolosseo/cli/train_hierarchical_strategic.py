"""Resumable neutral strategic PPO against one frozen strategic opponent.

This is a response-training primitive, not PSRO without matrix orchestration.
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from botcolosseo.agents.hierarchical_controller import HierarchicalController
from botcolosseo.agents.hierarchical_critic import StrategicActorCritic
from botcolosseo.agents.hierarchical_model import CommandExecutor, StrategicActor
from botcolosseo.data.hierarchical_demonstrations import load_command_episode
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv
from botcolosseo.training.hierarchical_collection import collect_strategic_episode
from botcolosseo.training.hierarchical_protocol import COMMAND_SCHEMA, ControlCondition
from botcolosseo.training.hierarchical_sampling import response_distribution
from botcolosseo.training.hierarchical_strategic_ppo import update_strategic_ppo


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--opponent", type=Path)
    parser.add_argument("--population", type=Path, nargs="+")
    parser.add_argument("--solution", type=Path)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--role", choices=("host", "opponent"), default="host")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--conditioned", action="store_true")
    args = parser.parse_args()
    if args.conditioned and (not args.initial or not args.opponent or args.population):
        raise ValueError(
            "Conditioned fine-tuning needs distilled initial and fixed opponent; "
            "no neutral meta-strategy reuse"
        )
    if bool(args.population) != bool(args.solution) or (args.population and args.opponent):
        raise ValueError("Population needs a solution and excludes single-opponent mode")
    if args.steps <= 0:
        raise ValueError("Positive learner decision budget required")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Preserve run; use --resume")
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    payload = torch.load(args.executor, map_location="cpu", weights_only=False)
    if payload["identity"]["command_schema"] != COMMAND_SCHEMA:
        raise ValueError("Executor command schema mismatch")
    example = load_command_episode(Path(next(iter(payload["identity"]["validation"]))))
    scenario = str(example["scenario_hash"])
    layouts = sorted(
        {
            int(load_command_episode(Path(path))["seed"]) % 128
            for path in payload["identity"]["train"]
        }
    )
    executor = CommandExecutor().to(args.device)
    executor.load_state_dict(payload["actor"])
    executor.requires_grad_(False)
    model = StrategicActorCritic(StrategicActor()).to(args.device)
    if args.initial:
        initial = torch.load(args.initial, map_location="cpu", weights_only=False)
        if initial["identity"]["executor"] != digest(args.executor):
            raise ValueError("Initial strategy belongs to another executor")
        model.actor.load_state_dict(initial["actor"])
    reference = (
        copy.deepcopy(model.actor).requires_grad_(False).eval() if args.conditioned else None
    )
    learning_rate = 1e-5 if args.conditioned else 1e-4
    conditions = [ControlCondition()]
    if args.conditioned:
        conditions += [
            ControlCondition(aggressive=1),
            ControlCondition(defensive=1),
            ControlCondition(explorer=1),
            ControlCondition(aggressive=1, defensive=1),
            ControlCondition(aggressive=1, explorer=1),
            ControlCondition(defensive=1, explorer=1),
        ]
    opponent = StrategicActor().to(args.device)
    if args.opponent is not None:
        other = torch.load(args.opponent, map_location="cpu", weights_only=False)
        if other["identity"]["executor"] != digest(args.executor):
            raise ValueError("Opponent belongs to a different executor window")
        opponent.load_state_dict(other["actor"])
    opponent.requires_grad_(False)
    opponents = [opponent]
    sigma = np.ones(1)
    if args.population:
        solution = json.loads(args.solution.read_text())
        if solution["identity"]["executor"] != digest(args.executor):
            raise ValueError("Solution belongs to another executor")
        opponent_role = "opponent" if args.role == "host" else "host"
        expected_population = solution["identity"].get(
            f"{opponent_role}_strategies", solution["identity"]["strategies"]
        )
        if [digest(path) for path in args.population] != expected_population:
            raise ValueError("Population ordering/hash differs from solved matrix")
        if solution["maximum_regret"] > 1e-3:
            raise ValueError("Unacceptable restricted-game residual")
        sigma = np.asarray(solution["opponent" if args.role == "host" else "host"])
        opponents = []
        for path in args.population:
            item = torch.load(path, map_location="cpu", weights_only=False)
            if item["identity"]["executor"] != digest(args.executor):
                raise ValueError("Population executor mismatch")
            actor = StrategicActor().to(args.device)
            actor.load_state_dict(item["actor"])
            actor.requires_grad_(False)
            opponents.append(actor)
    wins, draws, games = (np.zeros(len(opponents)) for _ in range(3))
    source_root = Path(__file__).parents[1]
    sources = [
        Path(__file__),
        *source_root.glob("training/hierarchical_*.py"),
        *source_root.glob("agents/hierarchical_*.py"),
    ]
    identity = {
        "executor": digest(args.executor),
        "scenario": scenario,
        "opponent": digest(args.opponent) if args.opponent else "seed1701-frozen-untrained",
        "role": args.role,
        "layouts": layouts,
        "command_schema": COMMAND_SCHEMA,
        "protocol": "neutral-own-bank150-high-sample-low-argmax-v1",
        "learning_rate": learning_rate,
        "gamma": 0.997,
        "sources": {str(path.relative_to(source_root)): digest(path) for path in sources},
    }
    if args.initial:
        identity["initial"] = digest(args.initial)
    if args.conditioned:
        identity.update(
            protocol="conditional-own-bank150-high-sample-low-argmax-reference-v1",
            conditions=[condition.as_tuple() for condition in conditions],
            condition_scope="same static condition for both roles; Hard only",
            reference_kl_weight=0.1,
            scope="fixed-opponent conditional task fine-tuning; not PSRO response improvement",
        )
    if args.population:
        identity.update(
            {
                "solution": digest(args.solution),
                "sigma": sigma.tolist(),
                "population": [digest(path) for path in args.population],
                "response_budget": args.steps,
                "curriculum": "0.7sigma-0.3pfsp-final25pure",
            }
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    steps = episode = 0
    if args.resume:
        saved = torch.load(args.output / "last.pt", map_location=args.device, weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Resume identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        steps, episode = saved["steps"], saved["episode"]
        if args.population:
            wins, draws, games = (np.asarray(saved[key]) for key in ("wins", "draws", "games"))
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = {key: value.clone() for key, value in executor.state_dict().items()}
    started = time.monotonic()
    initial_steps = steps
    while steps < args.steps:
        # Cross every condition with every training layout rather than assigning
        # each layout a permanent style label. Existing neutral sequence is unchanged.
        layout = layouts[(episode // len(conditions)) % len(layouts)]
        condition = conditions[episode % len(conditions)]
        probabilities = (
            response_distribution(
                sigma, wins, draws, games, completed_steps=steps, budget_steps=args.steps
            )
            if args.population
            else sigma
        )
        selected = int(
            np.random.default_rng(2701 + episode).choice(len(opponents), p=probabilities)
        )
        opponent = opponents[selected]
        controllers = {
            side: HierarchicalController(
                executor,
                model.actor if side == args.role else opponent,
                seed=1701 + 2 * episode + index,
            )
            for index, side in enumerate(("host", "opponent"))
        }
        env = SynchronousExtractionEnv(
            config_path=Path(
                "assets/scenarios/crystal_run_extraction_randomized/crystal_run_extraction_randomized.cfg"
            ),
            seed=episode,
            layout_variant=layout,
        )
        try:
            batch, report = collect_strategic_episode(
                env,
                controllers,
                model,
                learner_side=args.role,
                expected_scenario=scenario,
                condition=condition,
            )
        finally:
            env.close()
        update = update_strategic_ppo(
            model,
            optimizer,
            batch,
            reference=reference,
            reference_weight=0.1 if args.conditioned else 0,
        )
        own, other = report["payoffs"] if args.role == "host" else report["payoffs"][::-1]
        games[selected] += 1
        wins[selected] += own > other
        draws[selected] += own == other
        previous_steps = steps
        steps += report["learner_decisions"]
        episode += 1
        if any(not torch.equal(value, frozen[key]) for key, value in executor.state_dict().items()):
            raise RuntimeError("Shared executor changed during strategic training")
        saved = {
            "identity": identity,
            "model": model.state_dict(),
            "actor": model.actor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "steps": steps,
            "episode": episode,
            "budget": args.steps,
            "wins": wins.tolist(),
            "draws": draws.tolist(),
            "games": games.tolist(),
        }
        temporary = args.output / "last.tmp"
        torch.save(saved, temporary)
        temporary.replace(args.output / "last.pt")
        if steps // 25000 > previous_steps // 25000 or steps >= args.steps:
            torch.save(saved, args.output / f"candidate-{steps:07d}.pt")
        progress = {
            "steps": steps,
            "episode": episode,
            "budget": args.steps,
            "session_seconds": time.monotonic() - started,
            "session_steps": steps - initial_steps,
            "layout": layout,
            "condition": condition.as_tuple(),
            "last_episode": {key: value for key, value in report.items() if key != "timeline"},
            "update": update,
            "complete": steps >= args.steps,
            "opponent_index": selected,
            "sampling_probabilities": probabilities.tolist(),
            "games_by_opponent": games.tolist(),
        }
        temporary = args.output / "progress.tmp"
        temporary.write_text(json.dumps(progress, indent=2))
        temporary.replace(args.output / "progress.json")
        print(json.dumps(progress), flush=True)


if __name__ == "__main__":
    main()
