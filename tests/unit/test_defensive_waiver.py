import json
from pathlib import Path

import pytest

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.training.defensive_waiver import validate_defensive_data_admission


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _manifest(path: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "style": "defensive",
        "split": "train",
        "passed": False,
        "transitions": 50_000,
        "test_cases_accessed": False,
        "base_checkpoint_sha256": "a" * 64,
        "scenario_hash": "b" * 64,
        "min_denial_recovery_windows": 100,
        "window_counts": {"denial_recovery": 95},
        "gate": {
            "complete": True,
            "risk_transitions": True,
            "denial_recovery_windows": False,
            "opponent_coverage": True,
            "side_coverage": True,
            "no_risk_balance": True,
        },
    }
    _write(path, payload)
    return payload


def _waiver(path: Path, manifest_path: Path) -> None:
    _write(
        path,
        {
            "approved_on": "2026-07-23",
            "authorized_by": "project_owner",
            "base_checkpoint_sha256": "a" * 64,
            "data_manifest": "data/generated/m5/defensive/train-manifest.json",
            "data_manifest_sha256": sha256_file(manifest_path),
            "failed_gate": "denial_recovery_windows",
            "observed_denial_recovery_windows": 95,
            "required_denial_recovery_windows": 100,
            "scenario_hash": "b" * 64,
            "schema_version": 1,
            "stage": "m5-defensive-data-waiver",
            "style": "defensive",
            "test_cases_accessed": False,
            "transitions": 50_000,
        },
    )


def test_exact_defensive_waiver_admits_only_the_bound_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "train-manifest.json"
    waiver_path = tmp_path / "waiver.json"
    manifest = _manifest(manifest_path)
    _waiver(waiver_path, manifest_path)

    result = validate_defensive_data_admission(
        manifest, manifest_path=manifest_path, waiver_path=waiver_path
    )

    assert result == {
        "data_gate_passed": False,
        "data_waiver_applied": True,
        "data_waiver_sha256": sha256_file(waiver_path),
    }


@pytest.mark.parametrize("mutation", ("extra_gate", "changed_count", "test_access"))
def test_defensive_waiver_rejects_unapproved_manifest_changes(
    tmp_path: Path, mutation: str
) -> None:
    manifest_path = tmp_path / "train-manifest.json"
    waiver_path = tmp_path / "waiver.json"
    manifest = _manifest(manifest_path)
    _waiver(waiver_path, manifest_path)
    if mutation == "extra_gate":
        manifest["gate"]["complete"] = False  # type: ignore[index]
    elif mutation == "changed_count":
        manifest["window_counts"]["denial_recovery"] = 96  # type: ignore[index]
    else:
        manifest["test_cases_accessed"] = True

    with pytest.raises(ValueError):
        validate_defensive_data_admission(
            manifest, manifest_path=manifest_path, waiver_path=waiver_path
        )


def test_defensive_waiver_rejects_absent_or_stale_approval(tmp_path: Path) -> None:
    manifest_path = tmp_path / "train-manifest.json"
    waiver_path = tmp_path / "waiver.json"
    manifest = _manifest(manifest_path)
    _waiver(waiver_path, manifest_path)

    with pytest.raises(ValueError, match="explicit waiver"):
        validate_defensive_data_admission(
            manifest, manifest_path=manifest_path, waiver_path=None
        )
    manifest_path.write_text(manifest_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        validate_defensive_data_admission(
            manifest, manifest_path=manifest_path, waiver_path=waiver_path
        )
