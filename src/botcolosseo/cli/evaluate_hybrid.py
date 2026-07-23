from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import fields, replace
from pathlib import Path

import torch

from botcolosseo.agents.hybrid_config import (
    HybridPolicyConfig,
    load_hybrid_policy_config,
)
from botcolosseo.agents.hybrid_policy import (
    HybridEvaluationPolicy,
    build_hybrid_evaluation_policy,
)
from botcolosseo.agents.league_opponents import (
    CheckpointOpponentPolicy,
    OpponentSpec,
    sha256_file,
)
from botcolosseo.cli.evaluate_style import select_style_cases
from botcolosseo.cli.train_ppo import _atomic_json
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_types import DuelActorObservation, DuelPrivilegedState
from botcolosseo.evaluation.defensive import (
    DEFENSIVE_POLICIES,
    DefensiveEpisodeRecord,
    evaluate_defensive_records,
    run_defensive_episode,
)
from botcolosseo.evaluation.explorer import (
    EXPLORER_POLICIES,
    ExplorerEpisodeRecord,
    evaluate_explorer_records,
    run_explorer_episode,
)
from botcolosseo.evaluation.hybrid import (
    EpisodeRecord,
    HybridTelemetryRow,
    evaluate_hybrid_product,
)
from botcolosseo.evaluation.m2 import load_duel_cases
from botcolosseo.scenarios.duel_splits import DuelCase
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.training.bc import append_jsonl


class PublicBaseEvaluationPolicy:
    name = "strong_base"

    def __init__(self, policy: CheckpointOpponentPolicy) -> None:
        self._policy = policy

    def reset(self, *, seed: int) -> None:
        del seed
        self._policy.reset()

    def act(
        self,
        observation: DuelActorObservation,
        state: DuelPrivilegedState,
    ) -> MacroAction:
        del state
        return self._policy.act(observation)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a public-observation M5 hybrid style governor"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("configs/m2/validation.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs-per-opponent", type=int, default=1)
    parser.add_argument("--max-decisions", type=int, default=525)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if len(revision) != 40:
        raise ValueError("Could not bind hybrid evaluation to a Git revision")
    return revision


def _base_policy(config: HybridPolicyConfig, *, device: torch.device) -> PublicBaseEvaluationPolicy:
    spec = OpponentSpec(
        opponent_id="strong-base",
        kind="checkpoint",
        checkpoint=str(config.base_checkpoint),
        checkpoint_sha256=config.base_checkpoint_sha256,
        scenario_hash=config.scenario_hash,
        selection_evidence=f"m5-hybrid:{config.candidate_id}",
    )
    return PublicBaseEvaluationPolicy(CheckpointOpponentPolicy.load(spec, device=device))


def _hybrid_policy(
    config: HybridPolicyConfig,
    *,
    device: torch.device,
) -> HybridEvaluationPolicy:
    return build_hybrid_evaluation_policy(config, device=device)


def _load_records(path: Path, *, style: str) -> list[EpisodeRecord]:
    if not path.exists():
        return []
    record_type = DefensiveEpisodeRecord if style == "defensive" else ExplorerEpisodeRecord
    names = {field.name for field in fields(record_type)}
    return [
        record_type(**{name: payload[name] for name in names})  # type: ignore[arg-type]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for payload in (json.loads(line),)
    ]


def _load_telemetry(path: Path) -> list[HybridTelemetryRow]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        payload["base_action"] = MacroAction(payload["base_action"])
        payload["final_action"] = MacroAction(payload["final_action"])
        rows.append(HybridTelemetryRow(**payload))
    return rows


def _run_with_retries(runner, *, max_attempts: int) -> EpisodeRecord:
    for attempt in range(1, max_attempts + 1):
        try:
            return replace(runner(), environment_attempts=attempt)
        except RuntimeError as error:
            retriable = str(error) == "Duel respawn did not complete within the warm-up limit"
            if not retriable or attempt == max_attempts:
                raise
    raise AssertionError("unreachable")


def _validate_resume_prefix(
    records: list[EpisodeRecord],
    *,
    policies: tuple[str, str],
    selected: tuple[DuelCase, ...],
) -> None:
    expected = [
        (policy, case.opponent, case.pair_index, case.learner_side)
        for policy in policies
        for case in selected
    ]
    actual = [
        (row.policy, row.opponent, row.pair_index, row.learner_side) for row in records
    ]
    if actual != expected[: len(actual)]:
        raise ValueError("Hybrid episode ledger is not an ordered resume prefix")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(
        args.pairs_per_opponent,
        args.max_decisions,
        args.max_attempts,
        args.bootstrap_samples,
    ) <= 0:
        raise ValueError("M5 hybrid evaluation settings must be positive")
    root = Path(__file__).resolve().parents[3]
    config_path = _resolve(root, args.config)
    cases_path = _resolve(root, args.cases)
    output_dir = _resolve(root, args.output_dir)
    for path in (config_path, cases_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = load_hybrid_policy_config(config_path, root=root)
    if not config.base_checkpoint.is_file():
        raise FileNotFoundError(config.base_checkpoint)
    if sha256_file(config.base_checkpoint) != config.base_checkpoint_sha256:
        raise ValueError("Hybrid config Base checkpoint hash does not match")
    scenario_hash = json.loads(
        (root / "assets/scenarios/crystal_run/manifest.json").read_text(encoding="utf-8")
    )["wad_sha256"]
    if config.scenario_hash != scenario_hash:
        raise ValueError("Hybrid config scenario hash does not match the arena")
    cases = load_duel_cases(
        cases_path,
        expected_split="validation",
        pairs_per_opponent=50,
    )
    selected = select_style_cases(cases, pairs_per_opponent=args.pairs_per_opponent)
    policies = DEFENSIVE_POLICIES if config.style == "defensive" else EXPLORER_POLICIES
    identity = {
        "schema_version": 1,
        "stage": "m5-hybrid",
        "candidate_id": config.candidate_id,
        "style": config.style,
        "split": "validation",
        "base_checkpoint_sha256": config.base_checkpoint_sha256,
        "governor_config_sha256": config.config_sha256,
        "code_revision": _git_revision(root),
        "cases_sha256": sha256_file(cases_path),
        "selected_case_ids": [
            [case.opponent, case.pair_index, case.learner_side] for case in selected
        ],
        "scenario_hash": scenario_hash,
        "pairs_per_opponent": args.pairs_per_opponent,
        "max_decisions": args.max_decisions,
        "max_attempts": args.max_attempts,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "expected_episodes": len(policies) * len(selected),
        "test_cases_accessed": False,
    }
    if args.preflight:
        print(json.dumps({**identity, "preflight_passed": True}, indent=2, sort_keys=True))
        return 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.json"
    if run_path.exists():
        if json.loads(run_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("M5 hybrid evaluation resume identity does not match")
    else:
        _atomic_json(identity, run_path)
    episodes_path = output_dir / "episodes.jsonl"
    telemetry_path = output_dir / "telemetry.jsonl"
    records = _load_records(episodes_path, style=config.style)
    telemetry = _load_telemetry(telemetry_path)
    _validate_resume_prefix(records, policies=policies, selected=selected)
    graph = RegionGraph.from_yaml(root / "assets/scenarios/crystal_run/src/regions.yaml")
    arena_config = root / "assets/scenarios/crystal_run/crystal_run.cfg"
    base = _base_policy(config, device=device)
    hybrid = _hybrid_policy(config, device=device)
    completed_hybrid = sum(row.policy == config.style for row in records)
    for _ in range(completed_hybrid):
        hybrid.reset(seed=0)
    policy_map = {"strong_base": base, config.style: hybrid}
    total = int(identity["expected_episodes"])
    for policy_name in policies:
        for case in selected:
            key = (policy_name, case.opponent, case.pair_index, case.learner_side)
            completed = {
                (row.policy, row.opponent, row.pair_index, row.learner_side)
                for row in records
            }
            if key in completed:
                continue
            policy = policy_map[policy_name]

            def runner(
                selected_case: DuelCase = case,
                selected_policy=policy,
            ) -> EpisodeRecord:
                if config.style == "defensive":
                    return run_defensive_episode(
                        selected_case,
                        policy=selected_policy,
                        graph=graph,
                        config_path=arena_config,
                        max_decisions=args.max_decisions,
                    )
                return run_explorer_episode(
                    selected_case,
                    policy=selected_policy,
                    graph=graph,
                    config_path=arena_config,
                    max_decisions=args.max_decisions,
                )

            record = _run_with_retries(runner, max_attempts=args.max_attempts)
            append_jsonl(episodes_path, record.to_dict())
            records.append(record)
            if policy_name == config.style:
                rows = [
                    HybridTelemetryRow.from_policy(
                        style=config.style,
                        opponent=case.opponent,
                        pair_index=case.pair_index,
                        learner_side=case.learner_side,
                        telemetry=row,
                    )
                    for row in hybrid.drain_telemetry()
                ]
                for row in rows:
                    append_jsonl(telemetry_path, row.to_dict())
                telemetry.extend(rows)
            print(f"M5 {config.style} hybrid progress: {len(records)}/{total}", flush=True)
    if config.style == "defensive":
        legacy = evaluate_defensive_records(
            records,  # type: ignore[arg-type]
            expected_pairs_per_opponent=args.pairs_per_opponent,
            expected_scenario_hash=scenario_hash,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_samples=args.bootstrap_samples,
        )
    else:
        legacy = evaluate_explorer_records(
            records,  # type: ignore[arg-type]
            expected_pairs_per_opponent=args.pairs_per_opponent,
            expected_scenario_hash=scenario_hash,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_samples=args.bootstrap_samples,
        )
    maximum = config.governor.max_consecutive_interventions
    product = evaluate_hybrid_product(
        style=config.style,
        records=records,
        telemetry=telemetry,
        legacy_summary=legacy,
        max_consecutive_interventions=maximum,
    )
    payload = {
        "schema_version": 1,
        "stage": "m5-hybrid",
        "style": config.style,
        "candidate_id": config.candidate_id,
        "split": "validation",
        "product": product.to_dict(),
        "legacy_diagnostic": legacy.to_dict(),
        "test_cases_accessed": False,
    }
    summary_path = output_dir / "summary.json"
    _atomic_json(payload, summary_path)
    manifest = {
        **identity,
        "episodes": len(records),
        "episodes_sha256": sha256_file(episodes_path),
        "telemetry_sha256": sha256_file(telemetry_path),
        "summary_sha256": sha256_file(summary_path),
        "passed": product.passed,
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if product.passed else 1
