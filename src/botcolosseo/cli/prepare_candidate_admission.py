from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.cli.train_ppo import _atomic_json
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS
from botcolosseo.training.historical_pool import (
    AdmissionMetrics,
    PoolEntry,
    load_pool,
)
from botcolosseo.training.league_checkpoint import (
    LeagueCheckpointState,
    LeagueRunIdentity,
)


def _rate_grid(matrix: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = matrix.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Candidate matrix is missing {name}")
    return value


def _row(
    grid: Mapping[str, object], policy_id: str, axes: Sequence[str]
) -> dict[str, float]:
    raw = grid.get(policy_id)
    if not isinstance(raw, dict) or not set(axes).issubset(raw):
        raise ValueError("Candidate matrix payoff row is incomplete")
    values = {axis: raw[axis] for axis in axes}
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
        for value in values.values()
    ):
        raise ValueError("Candidate matrix contains invalid rates")
    return {axis: float(value) for axis, value in values.items()}


def derive_candidate_metrics(
    matrix: Mapping[str, object],
    *,
    candidate_id: str,
    active_ids: Sequence[str],
    script_ids: Sequence[str],
) -> dict[str, object]:
    active = tuple(active_ids)
    scripts = tuple(script_ids)
    if not active or len(scripts) != 5:
        raise ValueError("Candidate admission requires an active pool and five scripts")
    wins = _rate_grid(matrix, "win_rate")
    objectives = _rate_grid(matrix, "objective_rate")
    candidate_payoffs = _row(wins, candidate_id, active)
    active_payoffs = {policy_id: _row(wins, policy_id, active) for policy_id in active}
    script_wins = _row(wins, candidate_id, scripts)
    script_objectives = _row(objectives, candidate_id, scripts)
    active_policy_id = active[-1]
    return {
        "active_historical_worst_case": min(active_payoffs[active_policy_id].values()),
        "active_payoffs": active_payoffs,
        "candidate_historical_worst_case": min(candidate_payoffs.values()),
        "candidate_objective_rate": sum(script_objectives.values()) / len(scripts),
        "candidate_payoffs": candidate_payoffs,
        "candidate_script_average": sum(script_wins.values()) / len(scripts),
        "candidate_script_worst_case": min(script_wins.values()),
    }


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("M3 candidate artifacts must remain inside the repository") from error


def _object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def prepare_candidate_admission(
    *,
    artifact_root: Path,
    pool_path: Path,
    candidate_checkpoint: Path,
    candidate_id: str,
    matrix_dir: Path,
    output_entry: Path,
    output_metrics: Path,
    output_candidate_report: Path,
    source_git_commit: str,
    admitted_at_utc: str,
) -> tuple[PoolEntry, AdmissionMetrics]:
    root = artifact_root.expanduser().resolve()
    pool_path = pool_path.expanduser().resolve()
    candidate_checkpoint = candidate_checkpoint.expanduser().resolve()
    matrix_dir = matrix_dir.expanduser().resolve()
    outputs = tuple(
        path.expanduser().resolve()
        for path in (output_entry, output_metrics, output_candidate_report)
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("Candidate admission output already exists")
    if not source_git_commit or not admitted_at_utc:
        raise ValueError("Candidate provenance fields must be non-empty")
    pool = load_pool(pool_path, artifact_root=root)
    checkpoint_hash = sha256_file(candidate_checkpoint)
    checkpoint = torch.load(candidate_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Candidate is not an M3 league checkpoint")
    try:
        identity = LeagueRunIdentity(**checkpoint["identity"])
        state = LeagueCheckpointState(**checkpoint["state"])
    except (KeyError, TypeError) as error:
        raise ValueError("Candidate checkpoint metadata is invalid") from error
    if state.episodes % 2:
        raise ValueError("Candidate admission requires a paired episode boundary")
    if identity.pool_manifest_hash != pool.manifest_sha256:
        raise ValueError("Candidate checkpoint was not trained against the source pool")
    if identity.scenario_hash != pool.entries[0].scenario_hash:
        raise ValueError("Candidate scenario does not match the source pool")

    matrix_path = matrix_dir / "matrix.json"
    manifest_path = matrix_dir / "manifest.json"
    csv_path = matrix_dir / "crossplay.csv"
    matrix = _object(matrix_path)
    manifest = _object(manifest_path)
    active_ids = tuple(entry.policy_id for entry in pool.entries)
    expected_ids = {*active_ids, candidate_id, *DUEL_OPPONENTS}
    if (
        manifest.get("split") != "validation"
        or manifest.get("test_cases_accessed") is not False
        or manifest.get("pool_manifest_sha256") != pool.manifest_sha256
        or manifest.get("candidate_checkpoint_sha256") != checkpoint_hash
        or manifest.get("matrix_sha256") != sha256_file(matrix_path)
        or manifest.get("crossplay_csv_sha256") != sha256_file(csv_path)
        or manifest.get("protocol_inconsistencies") != 0
        or manifest.get("executed_rows") != 5 * len(expected_ids) * (len(expected_ids) + 1)
        or set(matrix.get("policy_ids", ())) != expected_ids
    ):
        raise ValueError("Candidate validation evidence is incomplete or mismatched")
    derived = derive_candidate_metrics(
        matrix,
        candidate_id=candidate_id,
        active_ids=active_ids,
        script_ids=DUEL_OPPONENTS,
    )
    metrics = AdmissionMetrics(
        integrity_ok=True,
        validation_complete=True,
        paired_side_swapped=True,
        protocol_inconsistencies=0,
        source_split="validation",
        candidate_script_average=float(derived["candidate_script_average"]),
        active_script_average=pool.entries[-1].script_average_win_rate,
        candidate_historical_worst_case=float(
            derived["candidate_historical_worst_case"]
        ),
        active_historical_worst_case=float(derived["active_historical_worst_case"]),
        candidate_payoffs=dict(derived["candidate_payoffs"]),
        active_payoffs=dict(derived["active_payoffs"]),
    )
    report_path = outputs[2]
    report_payload = {
        "schema_version": 1,
        "policy_id": candidate_id,
        "checkpoint": _relative(root, candidate_checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "environment_steps": state.environment_steps,
        "split": "validation",
        "test_cases_accessed": False,
        "integrity_passed": True,
        "rejection_reasons": [],
        "historical_worst_case_win_rate": metrics.candidate_historical_worst_case,
        "script_average_win_rate": metrics.candidate_script_average,
        "full_objective_rate": float(derived["candidate_objective_rate"]),
        "config_hash": identity.config_hash,
        "pool_manifest_sha256": pool.manifest_sha256,
        "payoff_report_sha256": sha256_file(matrix_path),
        "scenario_hash": identity.scenario_hash,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report_payload, report_path)
    entry = PoolEntry(
        policy_id=candidate_id,
        checkpoint=_relative(root, candidate_checkpoint),
        checkpoint_sha256=checkpoint_hash,
        scenario_hash=identity.scenario_hash,
        config_hash=identity.config_hash,
        source_git_commit=source_git_commit,
        parent_checkpoint_sha256=identity.base_checkpoint_sha256,
        environment_steps=state.environment_steps,
        admitted_at_utc=admitted_at_utc,
        validation_report=_relative(root, report_path),
        validation_report_sha256=sha256_file(report_path),
        script_average_win_rate=metrics.candidate_script_average,
        script_worst_case_win_rate=float(derived["candidate_script_worst_case"]),
        objective_rate=float(derived["candidate_objective_rate"]),
        payoff_by_policy=metrics.candidate_payoffs,
        anchor=False,
        admission_reason="validation_qualified_candidate",
    )
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    outputs[1].parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(asdict(entry), outputs[0])
    _atomic_json(asdict(metrics), outputs[1])
    return entry, metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare hash-bound M3 candidate admission evidence"
    )
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--output-entry", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--output-candidate-report", type=Path, required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--admitted-at-utc", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    entry, metrics = prepare_candidate_admission(
        artifact_root=root,
        pool_path=resolve(args.pool),
        candidate_checkpoint=resolve(args.candidate_checkpoint),
        candidate_id=args.candidate_id,
        matrix_dir=resolve(args.matrix_dir),
        output_entry=resolve(args.output_entry),
        output_metrics=resolve(args.output_metrics),
        output_candidate_report=resolve(args.output_candidate_report),
        source_git_commit=args.source_git_commit,
        admitted_at_utc=args.admitted_at_utc,
    )
    print(
        json.dumps(
            {
                "candidate_checkpoint_sha256": entry.checkpoint_sha256,
                "candidate_historical_worst_case": metrics.candidate_historical_worst_case,
                "candidate_policy_id": entry.policy_id,
                "candidate_script_average": metrics.candidate_script_average,
                "prepared": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
