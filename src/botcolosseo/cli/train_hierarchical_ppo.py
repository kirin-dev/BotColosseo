"""Episode-boundary resumable, fixed-reference command PPO pilot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from functools import partial
from pathlib import Path

import torch

from botcolosseo.agents.extraction_teachers import PrivilegedStrongExtractionTeacher
from botcolosseo.agents.hierarchical_controller import HierarchicalController
from botcolosseo.agents.hierarchical_model import (
    CommandActorCritic,
    CommandExecutor,
    StrategicActor,
)
from botcolosseo.cli.evaluate_hierarchical_executor import scheduled_command
from botcolosseo.data.hierarchical_demonstrations import episode_tensors, load_command_episode
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv
from botcolosseo.evaluation.hierarchical_role_paths import role_paths
from botcolosseo.training.hierarchical_executor_population import (
    FrozenCommandSelector,
    upgrade_pair,
)
from botcolosseo.training.hierarchical_ppo import update_command_ppo
from botcolosseo.training.hierarchical_protocol import require_neutral_matrix
from botcolosseo.training.hierarchical_rollout import collect_command_episode
from botcolosseo.training.hierarchical_schedule import replay_index, training_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=25000)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--task-weight", type=float, default=1.0)
    parser.add_argument("--population", type=Path, nargs="+")
    parser.add_argument("--solution", type=Path)
    args = parser.parse_args()
    if bool(args.population) != bool(args.solution):
        raise ValueError("Population upgrade requires both population and solution")
    if args.steps <= 0 or args.checkpoint_interval <= 0:
        raise ValueError("Budget must be positive")
    if not 0 < args.gamma <= 1:
        raise ValueError("Discount must be in (0,1]")
    if not 0 < args.task_weight <= 1:
        raise ValueError("Low-level task weight must be in (0,1]")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Preserving existing run")
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    initial = torch.load(args.initial, map_location="cpu", weights_only=False)
    identity = copy.deepcopy(initial["identity"])
    identity["ppo_initial"] = hashlib.sha256(args.initial.read_bytes()).hexdigest()
    population = None
    solution = None
    if args.population:
        solution = json.loads(args.solution.read_text())
        require_neutral_matrix(solution["identity"])
        if solution["identity"]["executor"] != identity["ppo_initial"]:
            raise ValueError("Upgrade must start from the matrix executor")
        if not 0 <= solution["maximum_regret"] <= 1e-3:
            raise ValueError("Unverified meta-strategy regret")
        paths = role_paths(solution["identity"], args.population)
        upgrade_pair(solution, 0)  # Validate both marginals before loading actors.
        population = {}
        for role, role_files in paths.items():
            if len(role_files) != len(solution[role]):
                raise ValueError("Role population and marginal length differ")
            population[role] = []
            for path in role_files:
                item = torch.load(path, map_location="cpu", weights_only=False)
                if item["identity"]["executor"] != identity["ppo_initial"]:
                    raise ValueError("Historical strategies use another executor version")
                high = StrategicActor().to(args.device).requires_grad_(False).eval()
                high.load_state_dict(item["actor"])
                population[role].append(high)
        identity["executor_upgrade"] = {
            "solution": hashlib.sha256(args.solution.read_bytes()).hexdigest(),
            "population": solution["identity"],
            "sampling": "half-uniform-half-meta-role-pairs-v1",
            "conditions": "Neutral/Hard",
        }
    identity["ppo_protocol"] = {
        "lr": 1e-5,
        "gamma": args.gamma,
        "task_weight": args.task_weight,
        "tf32": False,
        "reference_kl": 0.1,
        "replay_weight": 0.1,
        "replay_sampling": "shuffled-episode-cycles-v1",
        "schedule": "independent-balanced-command-contexts-v3",
        "checkpoint_interval": args.checkpoint_interval,
    }
    sources = [
        Path(__file__),
        Path("src/botcolosseo/training/hierarchical_ppo.py"),
        Path("src/botcolosseo/training/hierarchical_rollout.py"),
        Path("src/botcolosseo/training/hierarchical_schedule.py"),
        Path("src/botcolosseo/training/hierarchical_rewards.py"),
        Path("src/botcolosseo/agents/hierarchical_model.py"),
        Path("src/botcolosseo/agents/hierarchical_teacher.py"),
        Path("src/botcolosseo/agents/extraction_teachers.py"),
        Path("src/botcolosseo/cli/evaluate_hierarchical_executor.py"),
    ]
    if population is not None:
        sources.extend(
            [
                Path("src/botcolosseo/training/hierarchical_executor_population.py"),
                Path("src/botcolosseo/agents/hierarchical_controller.py"),
                Path("src/botcolosseo/evaluation/hierarchical_role_paths.py"),
            ]
        )
        identity["ppo_protocol"]["schedule"] = "frozen-population-fair-command-selector-v1"
    identity["ppo_sources"] = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    replay = []
    replay_paths = []
    for name, digest in identity["train"].items():
        path = Path(name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Replay source changed")
        episode = load_command_episode(path)
        if episode["valid"].any():
            replay.append(episode)
            replay_paths.append(str(path))
    if not replay:
        raise ValueError("No valid replay episodes")
    layouts = sorted({int(d["seed"]) % 128 for d in replay})
    scenarios = {str(d["scenario_hash"]) for d in replay}
    if len(scenarios) != 1:
        raise ValueError("Mixed replay scenarios")
    model = CommandActorCritic(CommandExecutor()).to(args.device)
    model.actor.load_state_dict(initial["actor"])
    reference = copy.deepcopy(model.actor).requires_grad_(False).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    steps = episode_index = 0
    elapsed = 0.0
    if args.resume:
        saved = torch.load(args.output / "last.pt", map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("PPO resume identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        steps, episode_index, elapsed = saved["steps"], saved["episodes"], saved["elapsed"]
        torch.set_rng_state(saved["cpu_rng"])
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state(saved["cuda_rng"], device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    initial_steps = steps
    while steps < args.steps:
        previous_steps = steps
        layout = layouts[(episode_index // 2) % len(layouts)]
        seed = layout + 128 * (episode_index // (2 * len(layouts)))
        side = "host" if episode_index % 2 == 0 else "opponent"
        other = "opponent" if side == "host" else "host"
        profile, command_seed = training_context(episode_index)
        command_selector = None
        opponent = PrivilegedStrongExtractionTeacher(side=other, layout_variant=layout)
        population_record = {}
        if population is not None:
            source, pair = upgrade_pair(solution, episode_index)
            command_selector = FrozenCommandSelector(
                population[side][pair[side]], seed=5701 + 2 * episode_index
            )
            opponent = HierarchicalController(
                model.actor, population[other][pair[other]], seed=5702 + 2 * episode_index
            )
            population_record = {"sampling_source": source, "strategy_indices": pair}
        env = SynchronousExtractionEnv(
            config_path=Path(
                "assets/scenarios/crystal_run_extraction_randomized/crystal_run_extraction_randomized.cfg"
            ),
            seed=seed,
            layout_variant=layout,
        )
        try:
            data, summary = collect_command_episode(
                model,
                env,
                side=side,
                layout_variant=layout,
                schedule=partial(scheduled_command, seed=command_seed, profile=profile),
                opponent=opponent,
                device=args.device,
                gamma=args.gamma,
                task_weight=args.task_weight,
                command_selector=command_selector,
            )
        finally:
            env.close()
        if summary["scenario_hash"] not in scenarios:
            raise ValueError("Live scenario mismatch")
        summary.update(
            seed=seed, layout=layout, side=side, profile=profile, command_seed=command_seed
        )
        if population is not None:
            summary.pop("profile")
            summary.pop("command_seed")
            summary.update(population_record)
        replay_choice = replay_index(episode_index, len(replay))
        summary["replay_path"] = replay_paths[replay_choice]
        demo = episode_tensors(replay[replay_choice], device=args.device)
        metrics = update_command_ppo(model, reference, optimizer, data, demo, gamma=args.gamma)
        steps += summary["learner_steps"]
        episode_index += 1
        wall = elapsed + time.monotonic() - started
        payload = {
            "actor": model.actor.state_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "identity": identity,
            "steps": steps,
            "episodes": episode_index,
            "elapsed": wall,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(args.device)
            if args.device.startswith("cuda")
            else None,
        }
        temporary = args.output / "last.tmp"
        torch.save(payload, temporary)
        temporary.replace(args.output / "last.pt")
        if (
            steps // args.checkpoint_interval > previous_steps // args.checkpoint_interval
            or steps >= args.steps
        ):
            candidate = args.output / f"candidate-{steps:07d}.pt"
            if candidate.exists():
                raise FileExistsError("Preserving existing candidate checkpoint")
            temporary = candidate.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(candidate)
        report = {
            "steps": steps,
            "episodes": episode_index,
            "elapsed": wall,
            "last_episode": summary,
            "update": metrics,
            "reward_components": {
                key: float(data[key].sum())
                for key in ("task_rewards", "potential_rewards", "success_rewards")
            },
            "session_steps_per_second": (steps - initial_steps) / (time.monotonic() - started),
        }
        (args.output / "progress.tmp").write_text(json.dumps(report, indent=2))
        (args.output / "progress.tmp").replace(args.output / "progress.json")
        print(json.dumps(report), flush=True)
    print("PPO budget completed at episode boundary", flush=True)


if __name__ == "__main__":
    main()
