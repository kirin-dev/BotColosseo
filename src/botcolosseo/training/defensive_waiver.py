from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from botcolosseo.agents.league_opponents import sha256_file

_GATES = {
    "complete",
    "risk_transitions",
    "denial_recovery_windows",
    "opponent_coverage",
    "side_coverage",
    "no_risk_balance",
}
_WAIVER_FIELDS = {
    "approved_on",
    "authorized_by",
    "base_checkpoint_sha256",
    "data_manifest",
    "data_manifest_sha256",
    "failed_gate",
    "observed_denial_recovery_windows",
    "required_denial_recovery_windows",
    "scenario_hash",
    "schema_version",
    "stage",
    "style",
    "test_cases_accessed",
    "transitions",
}


def validate_defensive_data_admission(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    waiver_path: Path | None,
) -> dict[str, object]:
    gates = manifest.get("gate")
    if (
        manifest.get("style") != "defensive"
        or manifest.get("split") != "train"
        or manifest.get("test_cases_accessed") is not False
        or manifest.get("transitions") != 50_000
        or not isinstance(gates, dict)
        or set(gates) != _GATES
        or any(type(value) is not bool for value in gates.values())
    ):
        raise ValueError("Defensive data manifest is not eligible for admission")
    if manifest.get("passed") is True:
        if not all(gates.values()):
            raise ValueError("Passing Defensive data manifest has a failed gate")
        if waiver_path is not None:
            raise ValueError("Passing Defensive data does not accept a waiver")
        return {
            "data_gate_passed": True,
            "data_waiver_applied": False,
            "data_waiver_sha256": None,
        }
    if waiver_path is None:
        raise ValueError("Failed Defensive data requires an explicit waiver")
    failed = {name for name, value in gates.items() if value is False}
    if failed != {"denial_recovery_windows"}:
        raise ValueError("Defensive waiver may cover only the window-count gate")
    if (
        manifest.get("min_denial_recovery_windows") != 100
        or manifest.get("window_counts", {}).get("denial_recovery") != 95
    ):
        raise ValueError("Defensive waiver counts do not match the approved shortfall")
    waiver = json.loads(waiver_path.read_text(encoding="utf-8"))
    if not isinstance(waiver, dict) or set(waiver) != _WAIVER_FIELDS:
        raise ValueError("Defensive data waiver does not match schema version 1")
    expected = {
        "approved_on": "2026-07-23",
        "authorized_by": "project_owner",
        "base_checkpoint_sha256": manifest.get("base_checkpoint_sha256"),
        "data_manifest": "data/generated/m5/defensive/train-manifest.json",
        "data_manifest_sha256": sha256_file(manifest_path),
        "failed_gate": "denial_recovery_windows",
        "observed_denial_recovery_windows": 95,
        "required_denial_recovery_windows": 100,
        "scenario_hash": manifest.get("scenario_hash"),
        "schema_version": 1,
        "stage": "m5-defensive-data-waiver",
        "style": "defensive",
        "test_cases_accessed": False,
        "transitions": 50_000,
    }
    if waiver != expected:
        raise ValueError("Defensive data waiver identity does not match")
    return {
        "data_gate_passed": False,
        "data_waiver_applied": True,
        "data_waiver_sha256": sha256_file(waiver_path),
    }
