from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from botcolosseo.agents.model import AsymmetricActorCritic
from botcolosseo.data.demonstrations import load_generation_cases, sha256_file
from botcolosseo.scenarios.duel_splits import DuelCase
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.training.bc import (
    append_jsonl,
    evaluate_closed_loop_episode,
    seed_everything,
)
from botcolosseo.training.curriculum import CurriculumPhase, OpponentCurriculum
from botcolosseo.training.duel_rollout import (
    DuelRolloutCollector,
    load_bc_actor_checkpoint,
)
from botcolosseo.training.ppo import ExcessiveKLError, PPOTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the recurrent M2 PPO agent")
    parser.add_argument("--config", type=Path, default=Path("configs/m2/ppo.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--environment-steps", type=int)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validation-pairs", type=int, default=0)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=100_000)
    return parser


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _provenance_hash(config: Path, train_cases: Path, bc_checkpoint: Path) -> str:
    digest = hashlib.sha256()
    digest.update(config.read_bytes())
    digest.update(train_cases.read_bytes())
    digest.update(sha256_file(bc_checkpoint).encode("ascii"))
    return digest.hexdigest()


def _run_hash(base_hash: str, *, target_steps: int, rollout_steps: int) -> str:
    payload = json.dumps(
        {"target_steps": target_steps, "rollout_steps": rollout_steps},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(f"{base_hash}:{payload}".encode()).hexdigest()


def _planned_updates(
    environment_steps: int,
    *,
    rollout_steps: int,
    sequence_length: int,
    minibatch_sequences: int,
    update_epochs: int,
) -> int:
    if min(
        environment_steps,
        rollout_steps,
        sequence_length,
        minibatch_sequences,
        update_epochs,
    ) <= 0:
        raise ValueError("PPO schedule values must be positive")
    remaining = environment_steps
    updates = 0
    while remaining:
        collected = min(rollout_steps, remaining)
        sequences = math.ceil(collected / sequence_length)
        updates += update_epochs * math.ceil(sequences / minibatch_sequences)
        remaining -= collected
    return updates


def _candidate_checkpoint_step(
    previous_steps: int, current_steps: int, *, interval: int, target: int
) -> int | None:
    if not 0 <= previous_steps < current_steps <= target or interval <= 0:
        raise ValueError("Invalid candidate checkpoint range")
    if current_steps == target or previous_steps // interval < current_steps // interval:
        return current_steps
    return None


def _copy_checkpoint(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _candidate_manifest(output_dir: Path) -> list[dict[str, object]]:
    return [
        {
            "checkpoint": path.name,
            "environment_steps": int(path.stem.split("-")[-1]),
            "sha256": sha256_file(path),
        }
        for path in sorted(output_dir.glob("candidate-*.pt"))
    ]


def _reconcile_metrics(path: Path, *, committed_environment_steps: int) -> None:
    if committed_environment_steps < 0:
        raise ValueError("committed_environment_steps must be nonnegative")
    if not path.exists():
        if committed_environment_steps:
            raise FileNotFoundError("PPO checkpoint exists without metrics")
        return
    kept: list[dict[str, object]] = []
    group_environment_steps = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            break
        if record.get("kind") == "rollout":
            group_environment_steps = int(record["environment_steps"])
        if group_environment_steps <= committed_environment_steps:
            kept.append(record)
    temporary = path.with_name(f".{path.name}.reconcile.tmp")
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in kept),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_history(path: Path) -> tuple[Counter[str], int, int, float, int]:
    events: Counter[str] = Counter()
    episodes = 0
    objectives = 0
    reward = 0.0
    kl_stops = 0
    if not path.exists():
        return events, episodes, objectives, reward, kl_stops
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") == "rollout":
            events.update(record["event_counts"])
        elif record.get("kind") == "episode":
            episodes += 1
            objectives += int(record["objective_completed"])
            reward += float(record["reward"])
        elif record.get("kind") == "kl_early_stop":
            kl_stops += 1
    return events, episodes, objectives, reward, kl_stops


def _curriculum(config: dict[str, object], train_cases: Path) -> OpponentCurriculum:
    phases = tuple(
        CurriculumPhase(
            int(item["start_environment_step"]), tuple(item["opponents"])
        )
        for item in config["curriculum"]
    )
    return OpponentCurriculum(
        load_generation_cases(train_cases, expected_split="train"),
        phases=phases,
        shaping_decay_steps=int(config["shaping_decay_steps"]),
    )


def _validation_cases(root: Path, pairs: int) -> tuple[DuelCase, ...]:
    if pairs < 0:
        raise ValueError("--validation-pairs must be nonnegative")
    if pairs == 0:
        return ()
    payload = json.loads((root / "configs/m2/validation.json").read_text())
    candidates = tuple(
        DuelCase(**item)
        for item in payload
        if item["opponent"] == "random_legal"
    )
    if any(case.split != "validation" for case in candidates):
        raise ValueError("PPO pilot may evaluate validation cases only")
    selected = candidates[: pairs * 2]
    if len(selected) != pairs * 2:
        raise ValueError("Not enough paired RandomLegal validation cases")
    for first, second in zip(selected[::2], selected[1::2], strict=True):
        if first.pair_index != second.pair_index or {
            first.learner_side,
            second.learner_side,
        } != {"host", "opponent"}:
            raise ValueError("RandomLegal validation cases are not side-swapped pairs")
    return selected


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir or root / config["output_dir"]
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists() and args.resume is None:
        raise FileExistsError(f"PPO output already exists: {metrics_path}")
    train_cases = root / config["train_cases"]
    bc_checkpoint = root / config["bc_checkpoint"]
    bc_summary_path = bc_checkpoint.parent / "summary.json"
    bc_summary = json.loads(bc_summary_path.read_text(encoding="utf-8"))
    bc_checkpoint_sha = sha256_file(bc_checkpoint)
    if bc_summary["checkpoint_sha256"] != bc_checkpoint_sha:
        raise ValueError("BC summary checkpoint hash does not match best.pt")
    scenario_hash = json.loads(
        (root / "assets/scenarios/crystal_run/manifest.json").read_text()
    )["wad_sha256"]
    if bc_summary["scenario_hash"] != scenario_hash:
        raise ValueError("BC checkpoint and current scenario do not match")
    target_steps = args.environment_steps or int(config["environment_steps"])
    stop_after = args.stop_after_steps or target_steps
    rollout_steps = args.rollout_steps or int(config["rollout_steps"])
    if args.checkpoint_interval_steps <= 0:
        raise ValueError("--checkpoint-interval-steps must be positive")
    config_hash = _run_hash(
        _provenance_hash(config_path, train_cases, bc_checkpoint),
        target_steps=target_steps,
        rollout_steps=rollout_steps,
    )
    if not 0 < stop_after <= target_steps:
        raise ValueError("--stop-after-steps must be in (0, environment-steps]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(int(config["seed"]))
    model = AsymmetricActorCritic().to(device)
    total_updates = _planned_updates(
        target_steps,
        rollout_steps=rollout_steps,
        sequence_length=int(config["sequence_length"]),
        minibatch_sequences=int(config["minibatch_sequences"]),
        update_epochs=int(config["update_epochs"]),
    )
    trainer = PPOTrainer.create(
        model,
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        total_updates=total_updates,
        gradient_clip=float(config["gradient_clip"]),
        policy_clip=float(config["policy_clip"]),
        value_clip=float(config["value_clip"]),
        value_coefficient=float(config["value_coefficient"]),
        entropy_coefficient=float(config["entropy_coefficient"]),
        max_kl=float(config["max_kl"]),
    )
    environment_steps = 0
    episode_index = 0
    if args.resume is None:
        bc_metadata = load_bc_actor_checkpoint(
            bc_checkpoint,
            model,
            expected_scenario_hash=scenario_hash,
            expected_checkpoint_sha=bc_checkpoint_sha,
        )
    else:
        metadata = trainer.load(
            args.resume,
            config_hash=config_hash,
            scenario_hash=scenario_hash,
            restore_rng=True,
        )
        environment_steps = metadata.counters["environment_steps"]
        episode_index = metadata.counters["episodes"]
        _reconcile_metrics(
            metrics_path, committed_environment_steps=environment_steps
        )
        bc_metadata = None
    if environment_steps > stop_after:
        raise ValueError("Resume checkpoint is beyond --stop-after-steps")
    (
        event_counts,
        logged_episodes,
        objectives,
        cumulative_reward,
        kl_early_stops,
    ) = _load_history(metrics_path)
    if logged_episodes != episode_index:
        raise ValueError("PPO metrics and checkpoint episode counters disagree")
    graph = RegionGraph.from_yaml(
        root / "assets/scenarios/crystal_run/src/regions.yaml"
    )
    collector = DuelRolloutCollector(
        model,
        curriculum=_curriculum(config, train_cases),
        graph=graph,
        device=device,
        config_path=root / "assets/scenarios/crystal_run/crystal_run.cfg",
        max_decisions=int(config["max_episode_decisions"]),
        episode_index=episode_index,
        gamma=float(config["gamma"]),
        gae_lambda=float(config["gae_lambda"]),
    )
    try:
        while environment_steps < stop_after:
            previous_environment_steps = environment_steps
            count = min(rollout_steps, stop_after - environment_steps)
            collection = collector.collect(
                steps=count, start_environment_step=environment_steps
            )
            environment_steps += collection.environment_steps
            event_counts.update(collection.event_counts)
            append_jsonl(
                metrics_path,
                {
                    "kind": "rollout",
                    "environment_steps": environment_steps,
                    "event_counts": collection.event_counts,
                    "episodes_completed": len(collection.episodes),
                },
            )
            for episode in collection.episodes:
                append_jsonl(metrics_path, {"kind": "episode", **asdict(episode)})
                objectives += int(episode.objective_completed)
                cumulative_reward += episode.reward
            stop_update = False
            for epoch in range(int(config["update_epochs"])):
                batches = collection.rollout.sequence_minibatches(
                    sequence_length=int(config["sequence_length"]),
                    burn_in=int(config["burn_in"]),
                    minibatch_sequences=int(config["minibatch_sequences"]),
                    seed=int(config["seed"]) + environment_steps,
                    epoch=epoch,
                )
                for batch in batches:
                    try:
                        metrics = trainer.train_step(batch)
                    except ExcessiveKLError as error:
                        kl_early_stops += 1
                        append_jsonl(
                            metrics_path,
                            {
                                "kind": "kl_early_stop",
                                "environment_steps": environment_steps,
                                "epoch": epoch,
                                "approximate_kl": error.approximate_kl,
                                "max_kl": error.max_kl,
                            },
                        )
                        stop_update = True
                        break
                    append_jsonl(
                        metrics_path,
                        {"kind": "train", "epoch": epoch, **asdict(metrics)},
                    )
                if stop_update:
                    break
            trainer.save(
                output_dir / "latest.pt",
                config_hash=config_hash,
                scenario_hash=scenario_hash,
                counters={
                    "environment_steps": environment_steps,
                    "episodes": collector.episode_index,
                },
            )
            candidate_step = _candidate_checkpoint_step(
                previous_environment_steps,
                environment_steps,
                interval=args.checkpoint_interval_steps,
                target=target_steps,
            )
            if candidate_step is not None:
                _copy_checkpoint(
                    output_dir / "latest.pt",
                    output_dir / f"candidate-{candidate_step:07d}.pt",
                )
            summary = {
                "bc_checkpoint": str(bc_checkpoint.relative_to(root)),
                "bc_checkpoint_sha256": bc_checkpoint_sha,
                "bc_updates": (
                    bc_metadata.counters["updates"]
                    if bc_metadata is not None
                    else bc_summary["best_update"]
                ),
                "checkpoint": _display_path(output_dir / "latest.pt", root),
                "checkpoint_sha256": sha256_file(output_dir / "latest.pt"),
                "candidate_checkpoints": _candidate_manifest(output_dir),
                "checkpoint_interval_steps": args.checkpoint_interval_steps,
                "completed": environment_steps == target_steps,
                "config": str(config_path.relative_to(root)),
                "config_hash": config_hash,
                "cumulative_episode_reward": cumulative_reward,
                "device": str(device),
                "environment_steps": environment_steps,
                "episode_count": collector.episode_index,
                "event_counts": dict(sorted(event_counts.items())),
                "kl_early_stop_count": kl_early_stops,
                "objective_completion_count": objectives,
                "scenario_hash": scenario_hash,
                "test_cases_accessed": False,
                "train_cases_hash": sha256_file(train_cases),
                "updates": trainer.updates,
            }
            _atomic_json(summary, output_dir / "summary.json")
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    finally:
        collector.close()
    if environment_steps == target_steps and args.validation_pairs:
        validation = [
            evaluate_closed_loop_episode(model.actor, root=root, case=case)
            for case in _validation_cases(root, args.validation_pairs)
        ]
        validation_summary = {
            "episode_count": len(validation),
            "objective_completion_count": sum(
                int(item["objective_completed"]) for item in validation
            ),
            "objective_rate": sum(
                int(item["objective_completed"]) for item in validation
            )
            / len(validation),
            "opponent": "random_legal",
            "paired_side_swaps": True,
            "split": "validation",
            "episodes": validation,
        }
        _atomic_json(validation_summary, output_dir / "validation-pilot.json")
        summary_path = output_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["validation_random_legal"] = validation_summary
        _atomic_json(summary, summary_path)
        print(json.dumps(validation_summary, indent=2, sort_keys=True), flush=True)
    return 0
