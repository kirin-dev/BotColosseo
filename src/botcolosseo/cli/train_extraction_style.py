from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from dataclasses import asdict, fields, replace
from pathlib import Path

import torch
import yaml

from botcolosseo.agents.checkpoint import load_model_weights_checkpoint
from botcolosseo.agents.extraction_model import (
    ExtractionResidualStyleActorCritic,
)
from botcolosseo.agents.extraction_teachers import StyledExtractionTeacher
from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.data.extraction_demonstrations import ExtractionCase
from botcolosseo.training.bc import append_jsonl, seed_everything
from botcolosseo.training.extraction_checkpoint import (
    load_extraction_strong_actor,
    load_extraction_strong_actor_critic,
)
from botcolosseo.training.extraction_pfsp import (
    ExtractionHistoricalOpponent,
    ExtractionPFSPSchedule,
)
from botcolosseo.training.extraction_rewards import (
    AggressiveExtractionRewardConfig,
    AggressiveExtractionRewardLedger,
    DefensiveExtractionRewardConfig,
    DefensiveExtractionRewardLedger,
    ExplorerExtractionRewardConfig,
    ExplorerExtractionRewardLedger,
)
from botcolosseo.training.extraction_rollout import (
    ExtractionRolloutCollector,
    PolicyExtractionOpponentController,
    RandomLegalExtractionOpponentController,
    ScriptExtractionOpponentController,
)
from botcolosseo.training.extraction_run_log import (
    reconcile_extraction_metrics,
)
from botcolosseo.training.ppo import ExcessiveKLError
from botcolosseo.training.style_ppo import StylePPOTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a learned Crystal Run: Extraction style adapter"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/extraction/styles.yaml"),
    )
    parser.add_argument(
        "--style",
        choices=("aggressive", "defensive", "explorer"),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--environment-steps", type=int)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument("--initialize-from", type=Path)
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
    if payload.get("split") != "train":
        raise ValueError("Style PPO may access train cases only")
    return tuple(ExtractionCase(**item) for item in payload["cases"])


def _config_hash(
    config: Path,
    cases: Path,
    base_checkpoint: Path,
    *,
    style: str,
    target_steps: int,
    rollout_steps: int,
) -> str:
    digest = hashlib.sha256()
    for path in (config, cases, base_checkpoint):
        digest.update(path.read_bytes())
    digest.update(
        json.dumps(
            {
                "style": style,
                "target_steps": target_steps,
                "rollout_steps": rollout_steps,
            },
            sort_keys=True,
        ).encode()
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
    updates = 0
    for start in range(0, environment_steps, rollout_steps):
        collected = min(rollout_steps, environment_steps - start)
        sequences = math.ceil(collected / sequence_length)
        updates += update_epochs * math.ceil(sequences / minibatch_sequences)
    return updates


def _resolved_style_reward_config(
    style: str,
    overrides: object = None,
):
    defaults = {
        "aggressive": AggressiveExtractionRewardConfig,
        "defensive": DefensiveExtractionRewardConfig,
        "explorer": ExplorerExtractionRewardConfig,
    }
    if style not in defaults:
        raise ValueError("Unsupported Extraction style")
    config = defaults[style]()
    if overrides is None:
        return config
    if not isinstance(overrides, dict):
        raise ValueError("Style reward overrides must be a mapping")
    known = {field.name: getattr(config, field.name) for field in fields(config)}
    if not set(overrides).issubset(known):
        raise ValueError("Style reward override field is unknown")
    for name, value in overrides.items():
        default = known[name]
        if isinstance(default, int):
            valid = isinstance(value, int) and not isinstance(value, bool)
        else:
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            )
        if not valid:
            raise ValueError("Style reward override value is invalid")
    return replace(config, **overrides)


def _style_reward_factory(style: str, scale: float, config):
    if style == "aggressive":
        return lambda side: AggressiveExtractionRewardLedger(
            config, learner_side=side, scale=scale
        )
    if style == "defensive":
        return lambda side: DefensiveExtractionRewardLedger(
            config, learner_side=side, scale=scale
        )
    if style == "explorer":
        return lambda side: ExplorerExtractionRewardLedger(
            config, learner_side=side, scale=scale
        )
    raise ValueError("Unsupported Extraction style")


def _initialize_style_weights(
    *,
    checkpoint: Path,
    model: torch.nn.Module,
    style: str,
    base_checkpoint_sha256: str,
    scenario_hash: str,
    root: Path,
) -> dict[str, str | int]:
    summary_path = checkpoint.parent / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Style initialization summary is missing: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint_sha256 = sha256_file(checkpoint)
    if (
        summary.get("style") != style
        or summary.get("scenario_hash") != scenario_hash
        or summary.get("base_checkpoint_sha256") != base_checkpoint_sha256
        or summary.get("checkpoint_sha256") != checkpoint_sha256
        or summary.get("completed") is not True
        or summary.get("test_cases_accessed") is not False
    ):
        raise ValueError("Style initialization summary provenance does not match")
    metadata = load_model_weights_checkpoint(
        checkpoint,
        model=model,
        expected_scenario_hash=scenario_hash,
    )
    if (
        summary.get("config_hash") != metadata.config_hash
        or summary.get("environment_steps")
        != metadata.counters.get("environment_steps")
        or summary.get("updates") != metadata.counters.get("updates")
    ):
        raise ValueError("Style initialization checkpoint provenance does not match")
    try:
        checkpoint_name = str(checkpoint.relative_to(root))
        summary_name = str(summary_path.relative_to(root))
    except ValueError as error:
        raise ValueError("Style initialization artifacts must be inside the project") from error
    return {
        "initialization_mode": "weights_only",
        "parent_checkpoint": checkpoint_name,
        "parent_checkpoint_sha256": checkpoint_sha256,
        "parent_config_hash": metadata.config_hash,
        "parent_environment_steps": metadata.counters["environment_steps"],
        "parent_summary": summary_name,
        "parent_updates": metadata.counters["updates"],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cases_path = root / config["train_cases"]
    base_checkpoint = args.base_checkpoint or root / config["base_checkpoint"]
    if not base_checkpoint.is_absolute():
        base_checkpoint = root / base_checkpoint
    base_sha256 = sha256_file(base_checkpoint)
    output_dir = args.output_dir or root / config["output_root"] / args.style
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists() and args.resume is None:
        raise FileExistsError(f"Style PPO output already exists: {metrics_path}")
    scenario_hash = json.loads(
        (
            root / "assets/scenarios/crystal_run_extraction/manifest.json"
        ).read_text(encoding="utf-8")
    )["wad_sha256"]
    target_steps = args.environment_steps or int(config["environment_steps"])
    stop_after = args.stop_after_steps or target_steps
    rollout_steps = args.rollout_steps or int(config["rollout_steps"])
    if not 0 < stop_after <= target_steps or rollout_steps <= 0:
        raise ValueError("Style PPO step schedule is invalid")
    config_hash = _config_hash(
        config_path,
        cases_path,
        base_checkpoint,
        style=args.style,
        target_steps=target_steps,
        rollout_steps=rollout_steps,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(int(config["seed"]))
    base, _ = load_extraction_strong_actor_critic(
        base_checkpoint,
        expected_scenario_hash=scenario_hash,
        expected_sha256=base_sha256,
        device=device,
    )
    model = ExtractionResidualStyleActorCritic(
        base,
        bottleneck=int(config["adapter_bottleneck"]),
        max_delta=float(config["max_delta"]),
    ).to(device)
    trainer = StylePPOTrainer.create(
        model,
        beta_kl=float(config["beta_kl"]),
        rho_residual=float(config["rho_residual"]),
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
    lineage: dict[str, str | int] = {}
    if args.resume is not None:
        resume = args.resume if args.resume.is_absolute() else root / args.resume
        metadata = trainer.load(
            resume,
            config_hash=config_hash,
            scenario_hash=scenario_hash,
            restore_rng=True,
        )
        environment_steps = metadata.counters["environment_steps"]
        episode_index = metadata.counters["episodes"]
        lineage = metadata.lineage
    elif args.initialize_from is not None:
        initialize_from = (
            args.initialize_from
            if args.initialize_from.is_absolute()
            else root / args.initialize_from
        )
        lineage = _initialize_style_weights(
            checkpoint=initialize_from,
            model=model,
            style=args.style,
            base_checkpoint_sha256=base_sha256,
            scenario_hash=scenario_hash,
            root=root,
        )
    history = reconcile_extraction_metrics(
        metrics_path,
        committed_environment_steps=environment_steps,
    )
    if history.episodes != episode_index:
        raise ValueError("Style PPO metrics and checkpoint episodes disagree")
    schedule = ExtractionPFSPSchedule(
        _load_cases(cases_path),
        shaping_decay_steps=int(config["shaping_decay_steps"]),
        master_seed=int(config["seed"]),
        history_probability=float(config["strong_opponent_probability"]),
    )
    schedule.add(
        ExtractionHistoricalOpponent(
            opponent_id="frozen-strong",
            checkpoint=base_checkpoint,
            checkpoint_sha256=base_sha256,
            environment_steps=1,
        )
    )
    strong_actor, _ = load_extraction_strong_actor(
        base_checkpoint,
        expected_scenario_hash=scenario_hash,
        expected_sha256=base_sha256,
        device=device,
    )
    style_reward_overrides = config.get("style_reward_overrides", {})
    if not isinstance(style_reward_overrides, dict):
        raise ValueError("Style reward overrides must be a mapping")
    resolved_style_reward_config = _resolved_style_reward_config(
        args.style,
        style_reward_overrides.get(args.style),
    )

    def opponent_factory(assignment, side):
        if assignment.opponent_kind == "script":
            if assignment.opponent_id == "random_legal":
                return RandomLegalExtractionOpponentController()
            return ScriptExtractionOpponentController(
                StyledExtractionTeacher(side=side, style=assignment.opponent_id)
            )
        return PolicyExtractionOpponentController(strong_actor, device=device)

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
        style_reward_factory=_style_reward_factory(
            args.style,
            float(config["style_reward_scale"][args.style]),
            resolved_style_reward_config,
        ),
    )
    events: Counter[str] = history.event_counts
    rewards: Counter[str] = history.reward_components
    kl_stops = history.kl_early_stops
    try:
        while environment_steps < stop_after:
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
                            "style_kl": trainer.last_style_kl,
                            "residual_magnitude": trainer.last_residual_magnitude,
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
                lineage=lineage,
            )
            interval = int(config["checkpoint_interval_steps"])
            if (
                environment_steps == target_steps
                or (environment_steps - collection.environment_steps) // interval
                < environment_steps // interval
            ):
                candidate = output_dir / f"candidate-{environment_steps:07d}.pt"
                candidate.parent.mkdir(parents=True, exist_ok=True)
                temporary = candidate.with_name(f".{candidate.name}.tmp")
                shutil.copyfile(latest, temporary)
                temporary.replace(candidate)
            summary = {
                "base_checkpoint": str(base_checkpoint.relative_to(root)),
                "base_checkpoint_sha256": base_sha256,
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
                "frozen_strong_actor": True,
                "frozen_strong_base": True,
                "kl_early_stop_count": kl_stops,
                "learned_residual_adapter": True,
                "lineage": lineage,
                "reward_components": dict(sorted(rewards.items())),
                "resolved_style_reward_config": asdict(
                    resolved_style_reward_config
                ),
                "scenario_hash": scenario_hash,
                "style": args.style,
                "style_reward_schema_version": (
                    4 if args.style == "defensive" else 1
                ),
                "test_cases_accessed": False,
                "train_cases_sha256": sha256_file(cases_path),
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "updates": trainer.updates,
            }
            _atomic_json(summary, output_dir / "summary.json")
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    finally:
        collector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
