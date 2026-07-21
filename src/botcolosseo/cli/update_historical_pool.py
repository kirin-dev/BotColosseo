from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.cli.train_ppo import _atomic_json
from botcolosseo.training.historical_pool import (
    AdmissionMetrics,
    PoolEntry,
    admission_decision,
    admit_candidate,
    load_pool,
    write_pool_atomic,
)


def _resolve(root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else root / path


def _object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _verify_entry(entry: PoolEntry, *, artifact_root: Path) -> None:
    for label, relative, expected_hash in (
        ("checkpoint", entry.checkpoint, entry.checkpoint_sha256),
        (
            "validation report",
            entry.validation_report,
            entry.validation_report_sha256,
        ),
    ):
        path = artifact_root / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Candidate {label} hash does not match")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply one validation-only M3 historical-pool admission"
    )
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-pool", type=Path, required=True)
    parser.add_argument("--decision-report", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[3]
    root = _resolve(project_root, args.artifact_root or Path("."))
    pool_path = _resolve(root, args.pool)
    entry_path = _resolve(root, args.entry)
    metrics_path = _resolve(root, args.metrics)
    output_pool = _resolve(root, args.output_pool)
    decision_report = _resolve(root, args.decision_report)
    if output_pool.exists() or decision_report.exists():
        raise FileExistsError("Historical-pool output evidence already exists")
    pool = load_pool(pool_path, artifact_root=root)
    try:
        entry = PoolEntry(**_object(entry_path))
        metrics = AdmissionMetrics(**_object(metrics_path))
    except TypeError as error:
        raise ValueError("Historical-pool admission fields do not match schema") from error
    _verify_entry(entry, artifact_root=root)
    decision = admission_decision(pool, entry, metrics)
    payload = {
        "schema_version": 1,
        "candidate_policy_id": entry.policy_id,
        "candidate_checkpoint_sha256": entry.checkpoint_sha256,
        "source_pool_manifest_sha256": pool.manifest_sha256,
        "pool_file_sha256": sha256_file(pool_path),
        "entry_file_sha256": sha256_file(entry_path),
        "metrics_file_sha256": sha256_file(metrics_path),
        "source_split": metrics.source_split,
        "test_cases_accessed": False,
        "hashes_verified": True,
        "eligible": decision.eligible,
        "reason": decision.reason,
        "replacement_policy_id": decision.replacement_policy_id,
        "new_pool_manifest_sha256": None,
    }
    if decision.eligible:
        updated = admit_candidate(pool, entry, metrics)
        output_pool.parent.mkdir(parents=True, exist_ok=True)
        write_pool_atomic(updated, output_pool)
        payload["new_pool_manifest_sha256"] = updated.manifest_sha256
    decision_report.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(payload, decision_report)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if decision.eligible else 2
