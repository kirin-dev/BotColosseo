from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from botcolosseo.agents.extraction_model import (
    create_extraction_actor_critic,
    freeze_extraction_actor_backbone,
)
from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.data.extraction_demonstrations import ExtractionCase
from botcolosseo.training.bc import append_jsonl, seed_everything
from botcolosseo.training.extraction_checkpoint import (
    load_extraction_bc_warm_start,
    load_extraction_strong_actor,
)
from botcolosseo.training.extraction_pfsp import (
    ExtractionHistoricalOpponent,
    ExtractionPFSPSchedule,
)
from botcolosseo.training.extraction_ppo import TeacherAnchoredPPOTrainer
from botcolosseo.training.extraction_rollout import (
    ExtractionRolloutCollector,
    PolicyExtractionOpponentController,
    ScriptExtractionOpponentController,
)
from botcolosseo.training.extraction_run_log import (
    reconcile_extraction_metrics,
)
from botcolosseo.training.ppo import ExcessiveKLError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Crystal Run: Extraction Strong with recurrent PPO"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/extraction/strong-ppo.yaml"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--environment-steps", type=int)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--bc-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-interval-steps", type=int)
    return parser


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_cases(path: Path) -> tuple[ExtractionCase, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "train" or payload.get("schema_version") != 1:
        raise ValueError("Strong PPO requires a schema-v1 train case manifest")
    cases = tuple(ExtractionCase(**item) for item in payload["cases"])
    if not cases or any(case.split != "train" for case in cases):
        raise ValueError("Strong PPO train cases are empty or contaminated")
    return cases


def _provenance_hash(paths: tuple[Path, ...], overrides: dict[str, int]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    digest.update(
        json.dumps(overrides, sort_keys=True, separators=(",", ":")).encode()
    )
    return digest.hexdigest()


def _planned_updates(
    environment_steps: int,
    *,
    rollout_steps: int,
    sequence_length: int,
    minibatch_sequences: int,
    update_epochs: int,
) -> int:
    remaining = environment_steps
    updates = 0
    while remaining:
        collected = min(rollout_steps, remaining)
        sequences = math.ceil(collected / sequence_length)
        updates += update_epochs * math.ceil(sequences / minibatch_sequences)
        remaining -= collected
    return updates


def _copy_checkpoint(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _candidate_step(
    previous: int, current: int, *, interval: int, target: int
) -> int | None:
    if current == target or previous // interval < current // interval:
        return current
    return None


def _candidate_manifest(output_dir: Path) -> list[dict[str, object]]:
    return [
        {
            "checkpoint": path.name,
            "environment_steps": int(path.stem.rsplit("-", 1)[1]),
            "sha256": sha256_file(path),
        }
        for path in sorted(output_dir.glob("candidate-*.pt"))
    ]


def _register_candidates(
    schedule: ExtractionPFSPSchedule,
    output_dir: Path,
) -> None:
    for item in _candidate_manifest(output_dir):
        schedule.add(
            ExtractionHistoricalOpponent(
                opponent_id=f"strong-{int(item['environment_steps']):07d}",
                checkpoint=output_dir / str(item["checkpoint"]),
                checkpoint_sha256=str(item["sha256"]),
                environment_steps=int(item["environment_steps"]),
            )
        )


def _load_pfsp_state(
    schedule: ExtractionPFSPSchedule,
    path: Path,
) -> None:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("test_cases_accessed") is not False:
            raise ValueError("PFSP state has invalid split provenance")
        schedule.load_state_dict(payload["outcomes"])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_cases_path = root / config["train_cases"]
    bc_checkpoint = args.bc_checkpoint or root / config["bc_checkpoint"]
    if not bc_checkpoint.is_absolute():
        bc_checkpoint = root / bc_checkpoint
    output_dir = args.output_dir or root / config["output_dir"]
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    metrics_path = output_dir / "metrics.jsonl"
    pfsp_state_path = output_dir / "pfsp-state.json"
    if metrics_path.exists() and args.resume is None:
        raise FileExistsError(f"Strong PPO output already exists: {metrics_path}")
    scenario_hash = json.loads(
        (
            root / "assets/scenarios/crystal_run_extraction/manifest.json"
        ).read_text(encoding="utf-8")
    )["wad_sha256"]
    bc_sha256 = sha256_file(bc_checkpoint)
    target_steps = args.environment_steps or int(config["environment_steps"])
    stop_after = args.stop_after_steps or target_steps
    rollout_steps = args.rollout_steps or int(config["rollout_steps"])
    checkpoint_interval = (
        args.checkpoint_interval_steps
        or int(config["checkpoint_interval_steps"])
    )
    if not 0 < stop_after <= target_steps or min(rollout_steps, checkpoint_interval) <= 0:
        raise ValueError("Strong PPO step schedule is invalid")
    config_hash = _provenance_hash(
        (config_path, train_cases_path, bc_checkpoint),
        {"environment_steps": target_steps, "rollout_steps": rollout_steps},
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(int(config["seed"]))
    model = create_extraction_actor_critic().to(device)
    freeze_actor_backbone = bool(config["freeze_actor_backbone"])
    if freeze_actor_backbone:
        freeze_extraction_actor_backbone(model)
    teacher_coefficient = float(config["teacher_auxiliary_coefficient"])
    trainer = TeacherAnchoredPPOTrainer.create(
        model,
        teacher_coefficient=teacher_coefficient,
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        total_updates=_planned_updates(
            target_steps,
            rollout_steps=rollout_steps,
            sequence_length=int(config["sequence_length"]),
            minibatch_sequences=int(config["minibatch_sequences"]),
            update_epochs=int(config["update_epochs"]),
        ),
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
        load_extraction_bc_warm_start(
            bc_checkpoint,
            model,
            expected_scenario_hash=scenario_hash,
            expected_sha256=bc_sha256,
        )
    else:
        resume = args.resume if args.resume.is_absolute() else root / args.resume
        metadata = trainer.load(
            resume,
            config_hash=config_hash,
            scenario_hash=scenario_hash,
            restore_rng=True,
        )
        environment_steps = metadata.counters["environment_steps"]
        episode_index = metadata.counters["episodes"]
    history = reconcile_extraction_metrics(
        metrics_path,
        committed_environment_steps=environment_steps,
    )
    if history.episodes != episode_index:
        raise ValueError("Strong PPO metrics and checkpoint episodes disagree")
    schedule = ExtractionPFSPSchedule(
        _load_cases(train_cases_path),
        shaping_decay_steps=int(config["shaping_decay_steps"]),
        master_seed=int(config["seed"]),
        history_probability=float(config["history_probability"]),
    )
    _register_candidates(schedule, output_dir)
    if args.resume is not None:
        _load_pfsp_state(schedule, pfsp_state_path)
    actor_cache: dict[str, torch.nn.Module] = {}

    def opponent_factory(assignment, side):
        if assignment.opponent_kind == "script":
            from botcolosseo.agents.extraction_teachers import StyledExtractionTeacher

            return ScriptExtractionOpponentController(
                StyledExtractionTeacher(side=side, style=assignment.opponent_id)
            )
        if assignment.opponent_id not in actor_cache:
            historical = {
                item.opponent_id: item for item in schedule.historical_opponents
            }[assignment.opponent_id]
            actor_cache[assignment.opponent_id], _ = load_extraction_strong_actor(
                historical.checkpoint,
                expected_scenario_hash=scenario_hash,
                expected_sha256=historical.checkpoint_sha256,
                device=device,
            )
        return PolicyExtractionOpponentController(
            actor_cache[assignment.opponent_id],
            device=device,
        )

    collector = ExtractionRolloutCollector(
        model,
        schedule=schedule,
        device=device,
        config_path=root
        / "assets/scenarios/crystal_run_extraction/crystal_run_extraction.cfg",
        max_decisions=int(config["max_episode_decisions"]),
        episode_index=episode_index,
        gamma=float(config["gamma"]),
        gae_lambda=float(config["gae_lambda"]),
        opponent_factory=opponent_factory,
        teacher_supervision=True,
    )
    events: Counter[str] = history.event_counts
    rewards: Counter[str] = history.reward_components
    kl_stops = history.kl_early_stops
    try:
        while environment_steps < stop_after:
            previous_steps = environment_steps
            collection = collector.collect(
                steps=min(rollout_steps, stop_after - environment_steps),
                start_environment_step=environment_steps,
            )
            environment_steps += collection.environment_steps
            events.update(collection.event_counts)
            rewards.update(collection.reward_components)
            append_jsonl(
                metrics_path,
                {
                    "kind": "rollout",
                    "environment_steps": environment_steps,
                    "episodes_completed": len(collection.episodes),
                    "event_counts": collection.event_counts,
                    "reward_components": collection.reward_components,
                },
            )
            for episode in collection.episodes:
                append_jsonl(metrics_path, {"kind": "episode", **asdict(episode)})
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
                        kl_stops += 1
                        append_jsonl(
                            metrics_path,
                            {
                                "kind": "kl_early_stop",
                                "environment_steps": environment_steps,
                                "approximate_kl": error.approximate_kl,
                            },
                        )
                        stop_update = True
                        break
                    append_jsonl(
                        metrics_path,
                        {
                            "kind": "train",
                            "environment_steps": environment_steps,
                            "epoch": epoch,
                            "teacher_loss": trainer.last_teacher_loss,
                            "teacher_agreement": trainer.last_teacher_agreement,
                            "supervised_tokens": trainer.last_supervised_tokens,
                            **asdict(metrics),
                        },
                    )
                if stop_update:
                    break
            latest = output_dir / "latest.pt"
            trainer.save(
                latest,
                config_hash=config_hash,
                scenario_hash=scenario_hash,
                counters={
                    "environment_steps": environment_steps,
                    "episodes": collector.episode_index,
                },
            )
            candidate_step = _candidate_step(
                previous_steps,
                environment_steps,
                interval=checkpoint_interval,
                target=target_steps,
            )
            if candidate_step is not None:
                _copy_checkpoint(
                    latest,
                    output_dir / f"candidate-{candidate_step:07d}.pt",
                )
                _register_candidates(schedule, output_dir)
            _atomic_json(
                {
                    "outcomes": schedule.state_dict(),
                    "test_cases_accessed": False,
                },
                pfsp_state_path,
            )
            summary = {
                "bc_checkpoint": str(bc_checkpoint.relative_to(root)),
                "bc_checkpoint_sha256": bc_sha256,
                "candidate_checkpoints": _candidate_manifest(output_dir),
                "checkpoint": str(latest.relative_to(root)),
                "checkpoint_sha256": sha256_file(latest),
                "completed": environment_steps == target_steps,
                "config": str(config_path.relative_to(root)),
                "config_hash": config_hash,
                "device": str(device),
                "environment_steps": environment_steps,
                "episode_count": collector.episode_index,
                "event_counts": dict(sorted(events.items())),
                "fair_actor_observation_only": True,
                "freeze_actor_backbone": freeze_actor_backbone,
                "kl_early_stop_count": kl_stops,
                "pfsp_win_rates": schedule.win_rates,
                "reward_components": dict(sorted(rewards.items())),
                "scenario_hash": scenario_hash,
                "test_cases_accessed": False,
                "teacher_auxiliary_coefficient": teacher_coefficient,
                "teacher_supervision": "privileged-strong-training-only",
                "train_cases_sha256": sha256_file(train_cases_path),
                "updates": trainer.updates,
            }
            _atomic_json(summary, output_dir / "summary.json")
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    finally:
        collector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
