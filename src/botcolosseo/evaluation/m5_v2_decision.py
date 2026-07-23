from __future__ import annotations

import json
from pathlib import Path

from botcolosseo.agents.league_opponents import sha256_file


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def decide_m5_v2_candidate(
    *,
    style: str,
    training_summary: Path,
    candidate: Path,
    smoke_dir: Path,
) -> dict[str, object]:
    if style not in ("defensive", "explorer"):
        raise ValueError("Unsupported M5 V2 style")
    training = _json(training_summary)
    summary_path = smoke_dir / "summary.json"
    manifest_path = smoke_dir / "manifest.json"
    summary = _json(summary_path)
    manifest = _json(manifest_path)
    candidate_hash = sha256_file(candidate)
    style_hashes = summary.get("checkpoint_sha256")
    if (
        training.get("style") != style
        or training.get("environment_steps") != 50_000
        or training.get("test_cases_accessed") is not False
        or not isinstance(style_hashes, dict)
        or style_hashes.get(style) != candidate_hash
        or manifest.get("summary_sha256") != sha256_file(summary_path)
        or manifest.get("episodes_sha256")
        != sha256_file(smoke_dir / "episodes.jsonl")
        or manifest.get("episodes") != 20
        or summary.get("complete") is not True
        or summary.get("protocol_inconsistencies") != 0
        or summary.get("test_cases_accessed") is not False
    ):
        raise ValueError("M5 V2 training/smoke identity is incomplete or inconsistent")
    gates = summary.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("M5 V2 smoke has no frozen gates")
    primary_name = (
        "protective_presence_delta"
        if style == "defensive"
        else "route_entropy_delta"
    )
    primary = summary.get(primary_name)
    if isinstance(primary, bool) or not isinstance(primary, (int, float)):
        raise ValueError("M5 V2 smoke has no finite primary style estimate")
    retention_passed = gates.get("skill_retention") is True
    protocol_passed = gates.get("protocol_clean") is True
    if summary.get("passed") is True:
        disposition = "select_50k"
        reasons = ["all frozen 20-episode smoke gates passed"]
    elif retention_passed and protocol_passed and float(primary) > 0.0:
        disposition = "continue_to_100k"
        reasons = [
            "skill retention and protocol gates passed",
            "primary style point estimate has the correct sign",
            "remaining frozen gates are inconclusive",
        ]
    else:
        disposition = "stop_50k"
        reasons = []
        if not retention_passed:
            reasons.append("skill retention gate failed")
        if not protocol_passed:
            reasons.append("protocol gate failed")
        if float(primary) <= 0.0:
            reasons.append("primary style point estimate has the wrong sign")
    try:
        candidate_label = str(candidate.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        candidate_label = str(candidate)
    return {
        "candidate_checkpoint": candidate_label,
        "candidate_checkpoint_sha256": candidate_hash,
        "disposition": disposition,
        "environment_steps": 50_000,
        "primary_metric": primary_name,
        "primary_point_estimate": float(primary),
        "reasons": reasons,
        "schema_version": 1,
        "smoke_manifest_sha256": sha256_file(manifest_path),
        "smoke_summary_sha256": sha256_file(summary_path),
        "style": style,
        "test_cases_accessed": False,
        "training_summary_sha256": sha256_file(training_summary),
    }
