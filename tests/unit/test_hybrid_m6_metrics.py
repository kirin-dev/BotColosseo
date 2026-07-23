from __future__ import annotations

from copy import deepcopy

import pytest

from botcolosseo.evaluation.hybrid_m6_metrics import (
    build_hybrid_m6_metric_payload,
)


def _inputs() -> dict[str, object]:
    artifacts = {
        "strong_base": "1" * 64,
        "aggressive": "2" * 64,
        "defensive": "3" * 64,
        "explorer": "4" * 64,
    }
    aggressive = {
        "stage": "m4",
        "passed": True,
        "complete": True,
        "test_cases_accessed": False,
        "gates": {"style": True},
        "checkpoint_sha256": {
            "strong_base": artifacts["strong_base"],
            "aggressive": artifacts["aggressive"],
        },
        "engagement_initiation_delta": 0.1,
        "policies": {"strong_base": {"win_rate": 0.87}},
    }

    def hybrid(style: str, **product: object) -> dict[str, object]:
        return {
            "stage": "m5-hybrid",
            "style": style,
            "test_cases_accessed": False,
            "product": {
                "passed": True,
                "complete": True,
                "gates": {"product": True},
                **product,
            },
        }

    retention = {
        style: {
            difficulty: {
                "aggregate": 0.9,
                "per_opponent": {"opponent": 0.8},
                "passed": True,
            }
            for difficulty in ("easy", "normal", "hard")
        }
        for style in ("defensive", "explorer")
    }
    difficulty = {
        "stage": "m5-hybrid-all-style-difficulty",
        "passed": True,
        "complete": True,
        "episodes": 1200,
        "test_cases_accessed": False,
        "gates": {"matrix": True},
        "retention": retention,
        "policy_artifact_sha256": artifacts,
        "policy_kinds": {
            "strong_base": "checkpoint",
            "aggressive": "checkpoint",
            "defensive": "hybrid_config",
            "explorer": "hybrid_config",
        },
        "scenario_hash": "5" * 64,
    }
    showcase = {
        "stage": "hybrid_product_showcase",
        "publication": True,
        "test_cases_accessed": False,
        "policy_artifact_sha256": artifacts,
        "scenario_hash": "5" * 64,
    }
    return {
        "aggressive": aggressive,
        "defensive": hybrid("defensive", skill_retention=0.959),
        "explorer": hybrid(
            "explorer",
            skill_retention=1.0,
            route_action_signature_distance=0.061,
        ),
        "difficulty": difficulty,
        "showcase": showcase,
        "upstream_sha256": {
            "aggressive": "6" * 64,
            "defensive": "7" * 64,
            "explorer": "8" * 64,
            "difficulty": "9" * 64,
            "showcase": "a" * 64,
        },
    }


def test_hybrid_m6_metrics_bind_product_matrix_and_showcase() -> None:
    payload = build_hybrid_m6_metric_payload(**_inputs())

    assert payload["passed"] is True
    assert payload["episodes"] == 1200
    assert payload["showcase_ready"] is False
    assert payload["headline_cards"][4] == {
        "label": "Min matrix retention",
        "value": "80.0%",
    }


def test_hybrid_m6_metrics_reject_showcase_artifact_drift() -> None:
    inputs = _inputs()
    showcase = deepcopy(inputs["showcase"])
    showcase["policy_artifact_sha256"]["explorer"] = "b" * 64
    inputs["showcase"] = showcase

    with pytest.raises(ValueError, match="showcase identity"):
        build_hybrid_m6_metric_payload(**inputs)
