from __future__ import annotations

import json
from pathlib import Path

import pytest

from botcolosseo.cli.resolve_extraction_showcase_artifact import (
    resolve_style_showcase_artifacts,
)
from botcolosseo.data.demonstrations import sha256_file


def _report(path: Path, policy: str, split: str = "validation") -> None:
    path.write_text(
        json.dumps(
            {
                "policy": policy,
                "metric_schema_version": 2,
                "split": split,
                "complete": True,
                "fair_actor_observation_only": True,
                "actor_privilege_violations": 0,
                "test_cases_accessed": False,
            }
        ),
        encoding="utf-8",
    )


def test_resolver_accepts_bound_pre_heldout_style_admission(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "runs/extraction/styles/explorer"
    directory.mkdir(parents=True)
    checkpoint = directory / "showcase.pt"
    checkpoint.write_bytes(b"explorer")
    validation = tmp_path / "explorer-validation.json"
    _report(validation, "explorer")
    relative = str(validation.relative_to(tmp_path))
    admission = {
        "admission_schema_version": 1,
        "admission_kind": "directional_showcase",
        "admission_rule_timing": "pre_heldout_product_rule",
        "policy": "explorer",
        "showcase_eligible": True,
        "research_gate_passed": False,
        "research_validation_failed_checks": ["style_ci_lower"],
        "original_heldout_gate_passed": True,
        "original_heldout_failed_checks": [],
        "research_failed_checks": ["style_ci_lower"],
        "original_heldout_checks": [
            {"name": "heldout_extraction_delta", "passed": True}
        ],
        "showcase_heldout_checks": [
            {"name": "showcase_heldout_extraction_delta", "passed": True}
        ],
        "direction_counts": {
            "positive_pairs": 33,
            "negative_pairs": 27,
            "new_showcase_chains": 23,
            "lost_showcase_chains": 13,
        },
        "actor_privilege_violations": 0,
        "test_cases_accessed": False,
        "showcase_checkpoint_sha256": sha256_file(checkpoint),
        "evidence": [relative],
        "evidence_sha256": {relative: sha256_file(validation)},
    }
    (directory / "showcase-admission.json").write_text(
        json.dumps(admission),
        encoding="utf-8",
    )

    resolved = resolve_style_showcase_artifacts(
        tmp_path,
        policy="explorer",
        directory=Path("runs/extraction/styles/explorer"),
    )

    assert resolved["mode"] == "directional_showcase"
    assert resolved["checkpoint"] == (
        "runs/extraction/styles/explorer/showcase.pt"
    )
    assert resolved["validation_report"] == "explorer-validation.json"


def test_resolver_rejects_failed_product_check(tmp_path: Path) -> None:
    directory = tmp_path / "runs/extraction/styles/explorer"
    directory.mkdir(parents=True)
    checkpoint = directory / "showcase.pt"
    checkpoint.write_bytes(b"explorer")
    validation = tmp_path / "explorer-validation.json"
    _report(validation, "explorer")
    relative = str(validation.relative_to(tmp_path))
    (directory / "showcase-admission.json").write_text(
        json.dumps(
            {
                "admission_schema_version": 1,
                "admission_kind": "directional_showcase",
                "admission_rule_timing": "pre_heldout_product_rule",
                "policy": "explorer",
                "showcase_eligible": True,
                "research_gate_passed": False,
                "research_validation_failed_checks": ["style_ci_lower"],
                "original_heldout_gate_passed": True,
                "original_heldout_failed_checks": [],
                "research_failed_checks": ["style_ci_lower"],
                "original_heldout_checks": [{"name": "original", "passed": True}],
                "showcase_heldout_checks": [
                    {"name": "relative", "passed": False}
                ],
                "direction_counts": {
                    "positive_pairs": 33,
                    "negative_pairs": 27,
                    "new_showcase_chains": 23,
                    "lost_showcase_chains": 13,
                },
                "actor_privilege_violations": 0,
                "test_cases_accessed": False,
                "showcase_checkpoint_sha256": sha256_file(checkpoint),
                "evidence": [relative],
                "evidence_sha256": {relative: sha256_file(validation)},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identity"):
        resolve_style_showcase_artifacts(
            tmp_path,
            policy="explorer",
            directory=Path("runs/extraction/styles/explorer"),
        )


def test_resolver_accepts_validation_demonstration_with_disclosed_failure(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "runs/extraction/styles/explorer"
    directory.mkdir(parents=True)
    checkpoint = directory / "showcase.pt"
    checkpoint.write_bytes(b"explorer")
    validation = tmp_path / "explorer-validation.json"
    heldout = tmp_path / "explorer-heldout.json"
    _report(validation, "explorer")
    _report(heldout, "explorer", "heldout")
    evidence = [
        str(validation.relative_to(tmp_path)),
        str(heldout.relative_to(tmp_path)),
    ]
    demonstration = {
        "demonstration_schema_version": 1,
        "evidence_tier": "validation_demonstration",
        "policy": "explorer",
        "product_demo_eligible": True,
        "research_gate_passed": False,
        "official_test_eligible": False,
        "research_failed_checks": [
            "style_ci_lower",
            "heldout_extraction_delta",
        ],
        "validation_failed_checks": ["style_ci_lower"],
        "heldout_gate_passed": False,
        "heldout_failed_checks": ["heldout_extraction_delta"],
        "heldout_checks": [
            {"name": "heldout_extraction_delta", "passed": False},
            {"name": "heldout_protocol_integrity", "passed": True},
        ],
        "actor_privilege_violations": 0,
        "test_cases_accessed": False,
        "showcase_checkpoint_sha256": sha256_file(checkpoint),
        "evidence": evidence,
        "evidence_sha256": {
            relative: sha256_file(tmp_path / relative) for relative in evidence
        },
    }
    (directory / "showcase-demonstration.json").write_text(
        json.dumps(demonstration),
        encoding="utf-8",
    )

    resolved = resolve_style_showcase_artifacts(
        tmp_path,
        policy="explorer",
        directory=Path("runs/extraction/styles/explorer"),
    )

    assert resolved["mode"] == "validation_demonstration"
    assert resolved["validation_report"] == "explorer-validation.json"
