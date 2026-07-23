from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from botcolosseo.agents.league_opponents import OpponentSpec, sha256_file
from botcolosseo.agents.model import AsymmetricActorCritic
from botcolosseo.agents.style_model import StyledActorCritic
from botcolosseo.cli.train_ppo import (
    _atomic_json,
    _copy_checkpoint,
    _planned_updates,
    _reconcile_metrics,
)
from botcolosseo.cli.train_ppo import (
    _candidate_checkpoint_step as _m2_candidate_checkpoint_step,
)
from botcolosseo.evaluation.m2_evidence_audit import (
    audit_official_evidence,
    audit_repository_provenance,
)
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS
from botcolosseo.scenarios.league_splits import load_league_cases
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.training.aggressive_reward import (
    AggressiveRewardConfig,
    AggressiveRewardLedger,
)
from botcolosseo.training.bc import append_jsonl, seed_everything
from botcolosseo.training.defensive_reward import (
    DefensiveRewardConfig,
    DefensiveRewardLedger,
)
from botcolosseo.training.explorer_reward import (
    ExplorerRewardConfig,
    ExplorerRewardLedger,
)
from botcolosseo.training.historical_pool import HistoricalPoolManifest, load_pool
from botcolosseo.training.league_checkpoint import (
    LeagueCheckpointState,
    LeagueRunIdentity,
    load_league_checkpoint,
    load_league_transition,
    save_league_checkpoint,
    warm_start_from_m2,
)
from botcolosseo.training.league_rollout import LeagueRolloutCollector
from botcolosseo.training.league_schedule import LeagueSchedule
from botcolosseo.training.ppo import ExcessiveKLError, PPOTrainer
from botcolosseo.training.style_distillation import load_style_distillation_checkpoint
from botcolosseo.training.style_ppo import StylePPOTrainer

_CONFIG_FIELDS = {
    "schema_version",
    "seed",
    "train_cases",
    "environment_steps",
    "rollout_steps",
    "candidate_interval_steps",
    "max_episode_decisions",
    "sequence_length",
    "burn_in",
    "minibatch_sequences",
    "update_epochs",
    "learning_rate",
    "weight_decay",
    "gamma",
    "gae_lambda",
    "policy_clip",
    "value_clip",
    "value_coefficient",
    "entropy_coefficient",
    "gradient_clip",
    "max_kl",
    "shaping_decay_steps",
}
_STYLE_FIELDS = {"name", "bottleneck", "lambda_style", "beta_kl", "reward"}


def _candidate_checkpoint_step(
    previous_steps: int, current_steps: int, *, interval: int, target: int
) -> int | None:
    return _m2_candidate_checkpoint_step(
        previous_steps, current_steps, interval=interval, target=target
    )


def _collection_pending(
    *,
    environment_steps: int,
    stop_after: int,
    episode_index: int,
    finish_paired_boundary: bool,
) -> bool:
    return environment_steps < stop_after or (
        finish_paired_boundary and episode_index % 2 != 0
    )


def _run_config_hash(
    config_file_hash: str, *, target_steps: int, rollout_steps: int
) -> str:
    if target_steps <= 0 or rollout_steps <= 0:
        raise ValueError("M3 run schedule must be positive")
    payload = json.dumps(
        {"rollout_steps": rollout_steps, "target_steps": target_steps},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(f"{config_file_hash}:{payload}".encode()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the recurrent M3 league agent")
    parser.add_argument("--config", type=Path, default=Path("configs/m3/league.yaml"))
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--payoffs", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume", type=Path)
    continuation.add_argument("--transition-from", type=Path)
    parser.add_argument("--environment-steps", type=int)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--finish-paired-boundary", action="store_true")
    base_authorization = parser.add_mutually_exclusive_group()
    base_authorization.add_argument("--allow-provisional-base", action="store_true")
    base_authorization.add_argument(
        "--allow-integrity-qualified-base", action="store_true"
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--style", choices=("aggressive", "defensive", "explorer"))
    parser.add_argument("--style-warm-start", type=Path)
    return parser


def _validate_config(config: dict[str, Any], *, style: str | None = None) -> None:
    expected = _CONFIG_FIELDS | ({"style"} if style is not None else set())
    if set(config) != expected:
        raise ValueError("M3 league config fields do not match the frozen schema")
    if config["schema_version"] != 1:
        raise ValueError("Unsupported M3 league config schema")
    train_cases = Path(str(config["train_cases"]))
    if train_cases.name != "train.json" or "configs/m3" not in train_cases.as_posix():
        raise ValueError("M3 league config must use the frozen train manifest")
    if int(config["candidate_interval_steps"]) != 200_000:
        raise ValueError("M3 candidate interval must remain 200000 steps")
    positive = (
        "environment_steps",
        "rollout_steps",
        "max_episode_decisions",
        "sequence_length",
        "minibatch_sequences",
        "update_epochs",
        "learning_rate",
        "gamma",
        "gae_lambda",
        "policy_clip",
        "value_clip",
        "gradient_clip",
        "max_kl",
        "shaping_decay_steps",
    )
    if any(float(config[name]) <= 0.0 for name in positive):
        raise ValueError("M3 league config contains nonpositive training values")
    if int(config["burn_in"]) < 0 or float(config["weight_decay"]) < 0.0:
        raise ValueError("M3 league config contains invalid regularization values")
    if style is not None:
        style_config = config["style"]
        if not isinstance(style_config, dict) or set(style_config) != _STYLE_FIELDS:
            raise ValueError("Style config fields do not match the frozen schema")
        if style_config["name"] != style:
            raise ValueError("Requested style does not match style config")
        if int(style_config["bottleneck"]) <= 0:
            raise ValueError("Style adapter bottleneck must be positive")
        if (
            float(style_config["lambda_style"]) < 0
            or float(style_config["beta_kl"]) < 0
        ):
            raise ValueError("Style coefficients must be nonnegative")
        reward = style_config["reward"]
        if not isinstance(reward, dict):
            raise ValueError("Style reward config must be a mapping")
        reward_config = {
            "aggressive": AggressiveRewardConfig,
            "defensive": DefensiveRewardConfig,
            "explorer": ExplorerRewardConfig,
        }[style]
        reward_config(**reward)


def _load_style_warm_start(
    path: Path,
    *,
    model: StyledActorCritic,
    expected_base_checkpoint_sha256: str,
    expected_scenario_hash: str,
    expected_style: str,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") == "style_distillation":
        return load_style_distillation_checkpoint(
            path,
            model=model,
            expected_base_checkpoint_sha256=expected_base_checkpoint_sha256,
            expected_scenario_hash=expected_scenario_hash,
            expected_style=expected_style,
        )
    expected = {
        "schema_version": 1,
        "kind": "style_interpolation",
        "style": expected_style,
        "base_checkpoint_sha256": expected_base_checkpoint_sha256,
        "scenario_hash": expected_scenario_hash,
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise ValueError("Style interpolation warm-start identity does not match")
    if payload.get("test_cases_accessed") is not False:
        report_path = path.with_suffix(".json")
        if not report_path.is_file():
            raise ValueError("Legacy style warm start has no zero-test-access report")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_identity = {
            "style": expected_style,
            "base_checkpoint_sha256": expected_base_checkpoint_sha256,
            "scenario_hash": expected_scenario_hash,
            "checkpoint_sha256": sha256_file(path),
            "test_cases_accessed": False,
        }
        if any(report.get(name) != value for name, value in report_identity.items()):
            raise ValueError("Legacy style warm-start report identity does not match")
    before = model.state_dict()
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("Style interpolation warm start has no model state")
    base_keys = tuple(name for name in before if name.startswith("base."))
    if any(
        name not in state or not torch.equal(before[name], state[name])
        for name in base_keys
    ):
        raise ValueError("Style interpolation warm start changed the frozen base")
    model.load_state_dict(state, strict=True)
    return {
        name: payload[name]
        for name in (
            "alpha",
            "base_checkpoint_sha256",
            "distilled_checkpoint_sha256",
            "interpolation_sha256",
            "scenario_hash",
        )
    }


def _style_reward_shaper(
    style: str,
    style_config: dict[str, Any],
    *,
    learner_side: str,
) -> AggressiveRewardLedger | DefensiveRewardLedger | ExplorerRewardLedger:
    scale = float(style_config["lambda_style"])
    reward = style_config["reward"]
    if style == "aggressive":
        return AggressiveRewardLedger(
            AggressiveRewardConfig(**reward),
            learner_side=learner_side,
            scale=scale,
        )
    if style == "defensive":
        return DefensiveRewardLedger(
            DefensiveRewardConfig(**reward),
            learner_side=learner_side,
            scale=scale,
        )
    if style == "explorer":
        return ExplorerRewardLedger(
            ExplorerRewardConfig(**reward),
            learner_side=learner_side,
            scale=scale,
        )
    raise ValueError("Unsupported style reward")


def _validate_base_authorization(
    base_checkpoint_sha256: str,
    audit_result: dict[str, object] | None,
    *,
    allow_provisional: bool,
    allow_integrity_qualified: bool = False,
) -> bool:
    if audit_result is None:
        if not allow_provisional:
            raise ValueError(
                "M2 base remains provisional; use --allow-provisional-base only for smoke"
            )
        return False
    hashes = audit_result.get("checkpoint_sha256")
    if not isinstance(hashes, dict) or hashes.get("ppo") != base_checkpoint_sha256:
        raise ValueError(
            "M2 audit checkpoint does not match the requested base checkpoint"
        )
    if audit_result.get("passed") is not True:
        if not (
            allow_integrity_qualified and audit_result.get("integrity_passed") is True
        ):
            raise ValueError("M2 capability gate did not pass")
        return False
    return True


def _load_payoff_report(path: Path, pool: HistoricalPoolManifest) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Invalid M3 payoff report schema")
    if payload.get("split") != "validation":
        raise ValueError("League training payoffs must come from validation")
    if payload.get("pool_manifest_sha256") != pool.manifest_sha256:
        raise ValueError("Payoff report does not match the active pool")
    win_rates = payload.get("win_rates")
    expected = {entry.policy_id for entry in pool.entries}
    if not isinstance(win_rates, dict) or set(win_rates) != expected:
        raise ValueError("Payoff report policy set does not match the active pool")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
        for value in win_rates.values()
    ):
        raise ValueError("Payoff report win rates must be finite probabilities")
    return {str(key): float(value) for key, value in win_rates.items()}


def _status_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _load_history(
    path: Path,
) -> tuple[
    Counter[str], Counter[str], Counter[str], Counter[str], int, int, float, int
]:
    events: Counter[str] = Counter()
    style_rewards: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    opponents: Counter[str] = Counter()
    episodes = 0
    objectives = 0
    reward = 0.0
    kl_stops = 0
    if not path.exists():
        return (
            events,
            style_rewards,
            sources,
            opponents,
            episodes,
            objectives,
            reward,
            kl_stops,
        )
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") == "rollout":
            events.update(record["event_counts"])
            style_rewards.update(record.get("style_reward_components", {}))
        elif record.get("kind") == "episode":
            episodes += 1
            objectives += int(record["objective_completed"])
            reward += float(record["reward"])
            sources[str(record["source"])] += 1
            opponents[str(record["opponent"])] += 1
        elif record.get("kind") == "kl_early_stop":
            kl_stops += 1
    return (
        events,
        style_rewards,
        sources,
        opponents,
        episodes,
        objectives,
        reward,
        kl_stops,
    )


def _candidate_manifest(run_dir: Path) -> list[dict[str, object]]:
    return [
        {
            "checkpoint": path.name,
            "environment_steps": int(path.stem.split("-")[-1]),
            "sha256": sha256_file(path),
        }
        for path in sorted(run_dir.glob("candidate-*.pt"))
    ]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else root / path


def _m2_audit_result(
    root: Path, *, integrity_only: bool = False
) -> dict[str, object] | None:
    report_dir = root / "reports/m2"
    targets = tuple(
        report_dir / name for name in ("episodes.csv", "summary.json", "manifest.json")
    )
    existing = tuple(path.exists() for path in targets)
    if not any(existing):
        return None
    if not all(existing):
        raise FileNotFoundError("Official M2 evidence is partially written")
    result = audit_official_evidence(
        report_dir, require_capability_pass=not integrity_only
    )
    return audit_repository_provenance(root, report_dir, result)


def _script_specs(scenario_hash: str) -> tuple[OpponentSpec, ...]:
    return tuple(
        OpponentSpec(
            opponent_id=name,
            kind="script",
            checkpoint=None,
            checkpoint_sha256=None,
            scenario_hash=scenario_hash,
            selection_evidence=f"builtin:{name}",
        )
        for name in DUEL_OPPONENTS
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.style_warm_start is not None and args.style is None:
        raise ValueError("--style-warm-start requires --style")
    if args.style_warm_start is not None and (
        args.resume is not None or args.transition_from is not None
    ):
        raise ValueError("Style warm start may only initialize a new run")
    root = _project_root()
    config_path = _resolve(root, args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("M3 league config must be a mapping")
    _validate_config(config, style=args.style)
    target_steps = args.environment_steps or int(config["environment_steps"])
    stop_after = args.stop_after_steps or target_steps
    rollout_steps = args.rollout_steps or int(config["rollout_steps"])
    if not 0 < stop_after <= target_steps or rollout_steps <= 0:
        raise ValueError("Invalid M3 stop or rollout step count")
    if args.finish_paired_boundary and stop_after >= target_steps:
        raise ValueError(
            "Paired-boundary padding requires room below the target budget"
        )
    base_checkpoint = _resolve(root, args.base_checkpoint)
    pool_path = _resolve(root, args.pool)
    payoff_path = _resolve(root, args.payoffs)
    run_dir = _resolve(root, args.run_dir)
    scenario_manifest = root / "assets/scenarios/crystal_run/manifest.json"
    scenario_payload = json.loads(scenario_manifest.read_text(encoding="utf-8"))
    scenario_hash = scenario_payload.get("wad_sha256")
    if not isinstance(scenario_hash, str):
        raise ValueError("Scenario manifest is missing wad_sha256")
    base_hash = sha256_file(base_checkpoint)
    style_warm_path = (
        None if args.style_warm_start is None else _resolve(root, args.style_warm_start)
    )
    style_warm_hash = None if style_warm_path is None else sha256_file(style_warm_path)
    if args.style is None:
        audit_result = _m2_audit_result(
            root, integrity_only=args.allow_integrity_qualified_base
        )
        base_promoted = _validate_base_authorization(
            base_hash,
            audit_result,
            allow_provisional=args.allow_provisional_base,
            allow_integrity_qualified=args.allow_integrity_qualified_base,
        )
    else:
        base_promoted = False
    pool = load_pool(pool_path, artifact_root=root)
    if args.style is None and pool.entries[0].checkpoint_sha256 != base_hash:
        raise ValueError("Historical pool anchor does not match the M2 base checkpoint")
    train_path = _resolve(root, Path(str(config["train_cases"])))
    train_cases = load_league_cases(
        train_path, expected_split="train", expected_pairs=250
    )
    win_rates = _load_payoff_report(payoff_path, pool)
    schedule = LeagueSchedule(
        cases=train_cases,
        scripts=_script_specs(scenario_hash),
        pool=pool,
        win_rates=win_rates,
        payoff_hash=sha256_file(payoff_path),
        master_seed=int(config["seed"]),
    )
    identity_config_hash = sha256_file(config_path)
    if style_warm_hash is not None:
        identity_config_hash = hashlib.sha256(
            f"{identity_config_hash}:{style_warm_hash}".encode()
        ).hexdigest()
    identity = LeagueRunIdentity(
        base_checkpoint_sha256=base_hash,
        config_hash=_run_config_hash(
            identity_config_hash,
            target_steps=target_steps,
            rollout_steps=rollout_steps,
        ),
        train_manifest_hash=sha256_file(train_path),
        pool_manifest_hash=pool.manifest_sha256,
        payoff_report_hash=sha256_file(payoff_path),
        scenario_hash=scenario_hash,
    )
    base_model = AsymmetricActorCritic()
    if args.style is None:
        warm_start_from_m2(
            base_checkpoint,
            base_model,
            expected_checkpoint_sha256=base_hash,
            expected_scenario_hash=scenario_hash,
        )
        model = base_model
    else:
        payload = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
        identity_payload = payload.get("identity")
        if payload.get("schema_version") != 1 or not isinstance(identity_payload, dict):
            raise ValueError("Style base must be an M3 league checkpoint")
        if identity_payload.get("scenario_hash") != scenario_hash:
            raise ValueError("Style base checkpoint scenario hash does not match")
        base_model.load_state_dict(payload["model"], strict=True)
        style_config = config["style"]
        model = StyledActorCritic.from_base(
            base_model, bottleneck=int(style_config["bottleneck"])
        )
        if style_warm_path is not None:
            _load_style_warm_start(
                style_warm_path,
                model=model,
                expected_base_checkpoint_sha256=base_hash,
                expected_scenario_hash=scenario_hash,
                expected_style=args.style,
            )
    preflight = {
        "base_checkpoint_sha256": base_hash,
        "base_promoted": base_promoted,
        "style": args.style,
        "style_warm_start": None if style_warm_path is None else str(style_warm_path),
        "style_warm_start_sha256": style_warm_hash,
        "style_base_capability_passed": None if args.style is None else False,
        "config_hash": identity.config_hash,
        "payoff_report_hash": identity.payoff_report_hash,
        "pool_manifest_sha256": pool.manifest_sha256,
        "preflight_passed": True,
        "run_dir": str(run_dir),
        "scenario_hash": scenario_hash,
        "test_cases_accessed": False,
        "train_case_count": len(train_cases),
        "train_manifest_hash": identity.train_manifest_hash,
    }
    if args.preflight:
        print(_status_json(preflight))
        return 0
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists() and args.resume is None and args.transition_from is None:
        raise FileExistsError(f"M3 league output already exists: {metrics_path}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(int(config["seed"]))
    model = model.to(device)
    total_updates = _planned_updates(
        target_steps,
        rollout_steps=rollout_steps,
        sequence_length=int(config["sequence_length"]),
        minibatch_sequences=int(config["minibatch_sequences"]),
        update_epochs=int(config["update_epochs"]),
    )
    trainer_factory = (
        PPOTrainer.create if args.style is None else StylePPOTrainer.create
    )
    trainer_kwargs = dict(
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
    if args.style is not None:
        trainer_kwargs["beta_kl"] = float(config["style"]["beta_kl"])
    trainer = trainer_factory(model, **trainer_kwargs)
    environment_steps = 0
    episode_index = 0
    transition_record: dict[str, object] | None = None
    if args.resume is not None:
        resume_path = _resolve(root, args.resume)
        state = load_league_checkpoint(
            resume_path,
            model=model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            expected_identity=identity,
            restore_rng=True,
        )
        trainer.updates = state.updates
        environment_steps = state.environment_steps
        episode_index = state.episodes
        _reconcile_metrics(metrics_path, committed_environment_steps=environment_steps)
    elif args.transition_from is not None:
        transition_path = _resolve(root, args.transition_from)
        transition_hash = sha256_file(transition_path)
        state, previous_identity = load_league_transition(
            transition_path,
            model=model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            next_identity=identity,
            restore_rng=True,
        )
        trainer.updates = state.updates
        environment_steps = state.environment_steps
        episode_index = state.episodes
        _reconcile_metrics(metrics_path, committed_environment_steps=environment_steps)
        archive_dir = run_dir / "transitions"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive = archive_dir / (
            f"parent-{environment_steps:07d}-{transition_hash[:12]}.pt"
        )
        if archive.exists() and sha256_file(archive) != transition_hash:
            raise ValueError("League transition archive hash conflicts with source")
        if not archive.exists():
            _copy_checkpoint(transition_path, archive)
        transition_record = {
            "archive_checkpoint": str(archive),
            "boundary_state": asdict(state),
            "next_identity": asdict(identity),
            "previous_identity": asdict(previous_identity),
            "schema_version": 1,
            "source_checkpoint": str(transition_path),
            "source_checkpoint_sha256": transition_hash,
        }
        _atomic_json(
            transition_record,
            archive_dir / f"transition-{environment_steps:07d}.json",
        )
    if environment_steps > stop_after and not (
        args.finish_paired_boundary and episode_index % 2 != 0
    ):
        raise ValueError("Resume checkpoint is beyond --stop-after-steps")
    (
        event_counts,
        style_reward_components,
        source_counts,
        opponent_counts,
        logged_episodes,
        objectives,
        cumulative_reward,
        kl_stops,
    ) = _load_history(metrics_path)
    if logged_episodes != episode_index:
        raise ValueError("M3 metrics and checkpoint episode counters disagree")
    graph = RegionGraph.from_yaml(
        root / "assets/scenarios/crystal_run/src/regions.yaml"
    )
    collector = LeagueRolloutCollector(
        model,
        schedule=schedule,
        graph=graph,
        device=device,
        shaping_decay_steps=int(config["shaping_decay_steps"]),
        config_path=root / "assets/scenarios/crystal_run/crystal_run.cfg",
        max_decisions=int(config["max_episode_decisions"]),
        episode_index=episode_index,
        gamma=float(config["gamma"]),
        gae_lambda=float(config["gae_lambda"]),
        reward_shaper_factory=None
        if args.style is None
        else lambda side: _style_reward_shaper(
            args.style,
            config["style"],
            learner_side=side,
        ),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        while _collection_pending(
            environment_steps=environment_steps,
            stop_after=stop_after,
            episode_index=collector.episode_index,
            finish_paired_boundary=args.finish_paired_boundary,
        ):
            previous_steps = environment_steps
            limit = stop_after if environment_steps < stop_after else target_steps
            count = min(rollout_steps, limit - environment_steps)
            if count <= 0:
                raise RuntimeError(
                    "M3 target budget ended before a paired episode boundary"
                )
            collection = collector.collect(
                steps=count, start_environment_step=environment_steps
            )
            environment_steps += collection.environment_steps
            event_counts.update(collection.event_counts)
            style_reward_components.update(getattr(collection, "reward_components", {}))
            append_jsonl(
                metrics_path,
                {
                    "kind": "rollout",
                    "environment_steps": environment_steps,
                    "event_counts": collection.event_counts,
                    "style_reward_components": getattr(
                        collection, "reward_components", {}
                    ),
                    "episodes_completed": len(collection.episodes),
                },
            )
            for episode in collection.episodes:
                append_jsonl(metrics_path, {"kind": "episode", **asdict(episode)})
                source_counts[episode.source] += 1
                opponent_counts[episode.opponent] += 1
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
                        kl_stops += 1
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
                    train_record = {
                        "kind": "train",
                        "epoch": epoch,
                        **asdict(metrics),
                    }
                    if args.style is not None:
                        train_record["style_base_kl"] = trainer.last_style_kl
                    append_jsonl(metrics_path, train_record)
                if stop_update:
                    break
            checkpoint_state = LeagueCheckpointState(
                environment_steps=environment_steps,
                updates=trainer.updates,
                episodes=collector.episode_index,
                next_pair_slot=collector.episode_index // 2,
            )
            latest = save_league_checkpoint(
                run_dir / "latest.pt",
                model=model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                identity=identity,
                state=checkpoint_state,
            )
            candidate_step = _candidate_checkpoint_step(
                previous_steps,
                environment_steps,
                interval=int(config["candidate_interval_steps"]),
                target=target_steps,
            )
            if candidate_step is not None:
                _copy_checkpoint(latest, run_dir / f"candidate-{candidate_step:07d}.pt")
            summary = {
                **preflight,
                "candidate_checkpoints": _candidate_manifest(run_dir),
                "checkpoint": str(latest),
                "checkpoint_sha256": sha256_file(latest),
                "completed": environment_steps == target_steps,
                "cumulative_episode_reward": cumulative_reward,
                "device": str(device),
                "environment_steps": environment_steps,
                "episode_count": collector.episode_index,
                "event_counts": dict(sorted(event_counts.items())),
                "style_reward_components": dict(
                    sorted(style_reward_components.items())
                ),
                "kl_early_stop_count": kl_stops,
                "objective_completion_count": objectives,
                "opponent_counts": dict(sorted(opponent_counts.items())),
                "opponent_source_counts": dict(sorted(source_counts.items())),
                "pfsp_probabilities": dict(schedule.pfsp_probabilities),
                "transition": transition_record,
                "updates": trainer.updates,
            }
            _atomic_json(summary, run_dir / "summary.json")
            print(_status_json(summary), flush=True)
    finally:
        collector.close()
    return 0
