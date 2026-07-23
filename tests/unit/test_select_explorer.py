import json
from pathlib import Path

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.cli.select_explorer import _candidate as build_candidate
from botcolosseo.cli.select_explorer import _compare
from botcolosseo.evaluation.explorer import ROUTE_ENTROPY_ESTIMATOR


def _rank_candidate(alpha: float, shift: float, retention: float) -> dict[str, object]:
    return {
        "alpha": alpha,
        "route_entropy_delta": shift,
        "skill_retention": retention,
    }


def test_explorer_ranking_prefers_clear_shift_then_retention_then_lower_alpha() -> None:
    assert _compare(
        _rank_candidate(0.75, 0.30, 0.86),
        _rank_candidate(0.25, 0.20, 1.0),
    ) < 0
    assert _compare(
        _rank_candidate(0.25, 0.30, 0.90),
        _rank_candidate(0.50, 0.27, 0.95),
    ) > 0
    assert _compare(
        _rank_candidate(0.25, 0.30, 0.95),
        _rank_candidate(0.50, 0.27, 0.95),
    ) < 0


def test_explorer_candidate_requires_the_frozen_estimator(tmp_path: Path) -> None:
    interpolation = tmp_path / "interpolation"
    smoke = tmp_path / "smoke"
    checkpoint = interpolation / "alpha-025.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_hash = sha256_file(checkpoint)
    report = {
        "alpha": 0.25,
        "style": "explorer",
        "checkpoint_sha256": checkpoint_hash,
        "test_cases_accessed": False,
    }
    summary = {
        "passed": True,
        "checkpoint_sha256": {"explorer": checkpoint_hash},
        "route_entropy_delta": 0.2,
        "skill_retention": 1.0,
        "test_cases_accessed": False,
    }
    manifest = {"passed": True, "episodes": 20, "test_cases_accessed": False}
    evaluation = smoke / "alpha-025"
    evaluation.mkdir(parents=True)
    for path, payload in (
        (interpolation / "alpha-025.json", report),
        (evaluation / "summary.json", summary),
        (evaluation / "manifest.json", manifest),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")

    assert build_candidate(
        0.25, interpolation_dir=interpolation, smoke_dir=smoke
    )["eligible"] is False
    summary["route_entropy_estimator"] = ROUTE_ENTROPY_ESTIMATOR
    manifest["route_entropy_estimator"] = ROUTE_ENTROPY_ESTIMATOR
    (evaluation / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (evaluation / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert build_candidate(
        0.25, interpolation_dir=interpolation, smoke_dir=smoke
    )["eligible"] is True
