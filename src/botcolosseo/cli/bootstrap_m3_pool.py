from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from botcolosseo.agents.checkpoint import CheckpointMetadata
from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.cli.train_ppo import _atomic_json
from botcolosseo.evaluation.m2_evidence_audit import (
    audit_official_evidence,
    audit_repository_provenance,
)
from botcolosseo.training.historical_pool import (
    HistoricalPoolManifest,
    PoolEntry,
    write_pool_atomic,
)


def _object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("M3 bootstrap artifacts must remain inside the repository") from error


def bootstrap_initial_pool(
    *,
    artifact_root: Path,
    checkpoint: Path,
    validation_evidence_dir: Path,
    output_pool: Path,
    output_payoffs: Path,
    policy_id: str,
    admitted_at_utc: str,
    audited_checkpoint_sha256: str,
) -> HistoricalPoolManifest:
    root = artifact_root.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    validation_evidence_dir = validation_evidence_dir.expanduser().resolve()
    output_pool = output_pool.expanduser().resolve()
    output_payoffs = output_payoffs.expanduser().resolve()
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != audited_checkpoint_sha256:
        raise ValueError("M2 audit does not authorize the bootstrap checkpoint")
    if output_pool.exists() or output_payoffs.exists():
        raise FileExistsError("M3 bootstrap outputs already exist")
    summary_path = validation_evidence_dir / "summary.json"
    manifest_path = validation_evidence_dir / "manifest.json"
    summary = _object(summary_path)
    manifest = _object(manifest_path)
    if (
        manifest.get("split") != "validation"
        or manifest.get("official") is not False
        or manifest.get("summary_sha256") != sha256_file(summary_path)
        or not isinstance(manifest.get("checkpoint_sha256"), dict)
        or manifest["checkpoint_sha256"].get("ppo") != checkpoint_hash  # type: ignore[union-attr]
        or summary.get("official") is not False
        or summary.get("protocol_inconsistencies") != 0
        or summary.get("artifact_inconsistencies") != 0
    ):
        raise ValueError("M2 anchor evidence is not clean validation-only evidence")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    try:
        metadata = CheckpointMetadata(**payload["metadata"])
    except (KeyError, TypeError) as error:
        raise ValueError("M2 anchor checkpoint metadata is invalid") from error
    if metadata.scenario_hash != manifest.get("scenario_hash"):
        raise ValueError("M2 anchor scenario provenance does not match")
    policies = summary.get("policies")
    if not isinstance(policies, dict) or not isinstance(policies.get("ppo"), dict):
        raise ValueError("M2 anchor validation summary is missing PPO metrics")
    ppo = policies["ppo"]
    opponents = ppo.get("opponents")
    if not isinstance(opponents, dict) or not opponents:
        raise ValueError("M2 anchor validation summary has no opponent metrics")
    try:
        average = float(ppo["wins"]["rate"])
        objective = float(ppo["objectives"]["rate"])
        worst = min(float(value["wins"]["rate"]) for value in opponents.values())
    except (KeyError, TypeError) as error:
        raise ValueError("M2 anchor validation rates are invalid") from error
    source_commit = manifest.get("git_commit")
    if not isinstance(source_commit, str) or not source_commit:
        raise ValueError("M2 anchor validation manifest is missing its commit")
    entry = PoolEntry(
        policy_id=policy_id,
        checkpoint=_relative(root, checkpoint),
        checkpoint_sha256=checkpoint_hash,
        scenario_hash=metadata.scenario_hash,
        config_hash=metadata.config_hash,
        source_git_commit=source_commit,
        parent_checkpoint_sha256=checkpoint_hash,
        environment_steps=int(metadata.counters.get("environment_steps", 0)),
        admitted_at_utc=admitted_at_utc,
        validation_report=_relative(root, summary_path),
        validation_report_sha256=sha256_file(summary_path),
        script_average_win_rate=average,
        script_worst_case_win_rate=worst,
        objective_rate=objective,
        payoff_by_policy={policy_id: 0.5},
        anchor=True,
        admission_reason="audited_m2_anchor",
    )
    pool = HistoricalPoolManifest(
        schema_version=1,
        pool_version=0,
        parent_manifest_sha256=None,
        created_at_utc=admitted_at_utc,
        entries=(entry,),
    )
    output_pool.parent.mkdir(parents=True, exist_ok=True)
    output_payoffs.parent.mkdir(parents=True, exist_ok=True)
    write_pool_atomic(pool, output_pool)
    _atomic_json(
        {
            "schema_version": 1,
            "split": "validation",
            "pool_manifest_sha256": pool.manifest_sha256,
            "win_rates": {policy_id: 0.5},
        },
        output_payoffs,
    )
    return pool


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap the audited M3 anchor pool")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-evidence-dir", type=Path, required=True)
    parser.add_argument("--output-pool", type=Path, required=True)
    parser.add_argument("--output-payoffs", type=Path, required=True)
    parser.add_argument("--policy-id", default="m2-anchor")
    parser.add_argument("--admitted-at-utc", required=True)
    parser.add_argument("--m2-report-dir", type=Path, default=Path("reports/m2"))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    report_dir = resolve(args.m2_report_dir)
    audited = audit_official_evidence(report_dir)
    audited = audit_repository_provenance(root, report_dir, audited)
    hashes = audited.get("checkpoint_sha256")
    if not isinstance(hashes, dict) or not isinstance(hashes.get("ppo"), str):
        raise ValueError("M2 audit did not report an authorized PPO checkpoint")
    pool = bootstrap_initial_pool(
        artifact_root=root,
        checkpoint=resolve(args.base_checkpoint),
        validation_evidence_dir=resolve(args.validation_evidence_dir),
        output_pool=resolve(args.output_pool),
        output_payoffs=resolve(args.output_payoffs),
        policy_id=args.policy_id,
        admitted_at_utc=args.admitted_at_utc,
        audited_checkpoint_sha256=hashes["ppo"],
    )
    print(
        json.dumps(
            {
                "checkpoint_sha256": pool.entries[0].checkpoint_sha256,
                "pool_manifest_sha256": pool.manifest_sha256,
                "policy_id": pool.entries[0].policy_id,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
