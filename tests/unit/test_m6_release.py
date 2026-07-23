from __future__ import annotations

from copy import deepcopy

import pytest

from botcolosseo.evaluation.m6_release import build_m6_showcase_metric_payload


def _inputs() -> dict[str, object]:
    base = "1" * 64
    aggressive = "2" * 64
    defensive_hash = "3" * 64
    explorer_hash = "4" * 64
    m4 = {
        "stage": "m4",
        "split": "validation",
        "passed": True,
        "complete": True,
        "episodes": 200,
        "test_cases_accessed": False,
        "checkpoint_sha256": {
            "strong_base": base,
            "aggressive": aggressive,
        },
        "gates": {"style": True, "retention": True},
        "skill_retention": 1.0,
        "engagement_initiation_delta": 0.10,
        "policies": {"strong_base": {"win_rate": 0.87}},
    }
    m4_showcase = {
        "schema_version": 1,
        "stage": "m4",
        "split": "validation",
        "passed": True,
        "checkpoint_sha256": m4["checkpoint_sha256"],
        "case_contrast_scores": {"case": 2},
        "decision_contrast_scores": {"case": [0, 1, 0]},
    }
    defensive = {
        "stage": "m5-defensive",
        "split": "validation",
        "passed": True,
        "complete": True,
        "episodes": 200,
        "test_cases_accessed": False,
        "checkpoint_sha256": {
            "strong_base": base,
            "defensive": defensive_hash,
        },
        "gates": {"style": True, "retention": True},
        "skill_retention": 0.91,
        "protective_presence_delta": 0.08,
    }
    explorer = {
        "stage": "m5-explorer",
        "split": "validation",
        "passed": True,
        "complete": True,
        "episodes": 200,
        "test_cases_accessed": False,
        "checkpoint_sha256": {
            "strong_base": base,
            "explorer": explorer_hash,
        },
        "gates": {"style": True, "retention": True},
        "skill_retention": 0.90,
        "route_entropy_delta": 0.12,
    }
    difficulty = {
        "stage": "m5-difficulty",
        "split": "validation",
        "passed": True,
        "complete": True,
        "episodes": 1200,
        "test_cases_accessed": False,
        "checkpoint_sha256": {
            "strong_base": base,
            "aggressive": aggressive,
            "defensive": defensive_hash,
            "explorer": explorer_hash,
        },
        "gates": {"monotonic": True, "style": True},
    }
    return {
        "m4": m4,
        "m4_showcase": m4_showcase,
        "defensive": defensive,
        "explorer": explorer,
        "difficulty": difficulty,
        "upstream_sha256": {
            "m4": "5" * 64,
            "defensive": "6" * 64,
            "explorer": "7" * 64,
            "difficulty": "8" * 64,
        },
    }


def test_m6_payload_binds_all_passing_upstreams() -> None:
    payload = build_m6_showcase_metric_payload(**_inputs())

    assert payload["passed"] is True
    assert payload["episodes"] == 1800
    assert payload["checkpoint_sha256"]["defensive"] == "3" * 64
    assert payload["headline_cards"][4] == {
        "label": "Min retention",
        "value": "90.0%",
    }
    assert payload["case_contrast_scores"] == {"case": 2.0}
    assert payload["test_cases_accessed"] is False


def test_m6_payload_rejects_failed_style_gate() -> None:
    inputs = _inputs()
    defensive = deepcopy(inputs["defensive"])
    defensive["passed"] = False
    inputs["defensive"] = defensive

    with pytest.raises(ValueError, match="m5-defensive"):
        build_m6_showcase_metric_payload(**inputs)


def test_m6_payload_rejects_checkpoint_drift() -> None:
    inputs = _inputs()
    difficulty = deepcopy(inputs["difficulty"])
    difficulty["checkpoint_sha256"]["explorer"] = "9" * 64
    inputs["difficulty"] = difficulty

    with pytest.raises(ValueError, match="inconsistent"):
        build_m6_showcase_metric_payload(**inputs)


def test_m6_payload_rejects_nonpositive_style_shift() -> None:
    inputs = _inputs()
    explorer = deepcopy(inputs["explorer"])
    explorer["route_entropy_delta"] = 0.0
    inputs["explorer"] = explorer

    with pytest.raises(ValueError, match="must be positive"):
        build_m6_showcase_metric_payload(**inputs)
