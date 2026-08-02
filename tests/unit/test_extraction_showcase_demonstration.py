from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from botcolosseo.cli.create_extraction_showcase_demonstration import (
    build_validation_demonstration,
)
from botcolosseo.data.demonstrations import sha256_file
from tests.unit.test_extraction_gates import episodes
from tests.unit.test_extraction_showcase_admission import _write_report


def _evidence(tmp_path: Path) -> dict[str, Path | str]:
    strong_checkpoint = tmp_path / "strong.pt"
    explorer_checkpoint = tmp_path / "explorer.pt"
    strong_checkpoint.write_bytes(b"strong")
    explorer_checkpoint.write_bytes(b"explorer")
    strong_sha256 = sha256_file(strong_checkpoint)
    paths = {
        "strong_validation_path": tmp_path / "strong-validation.json",
        "validation_path": tmp_path / "explorer-validation.json",
        "strong_heldout_path": tmp_path / "strong-heldout.json",
        "heldout_path": tmp_path / "explorer-heldout.json",
    }
    strong_validation = episodes(240)
    explorer_validation = list(strong_validation)
    for index in range(140):
        explorer_validation[index] = replace(
            explorer_validation[index],
            meaningful_loot_regions=3,
            backpack_upgrades=1,
            upgrade_to_extraction_conversions=1,
        )
    for index in range(140, 240):
        explorer_validation[index] = replace(
            explorer_validation[index],
            meaningful_loot_regions=0,
        )
    strong_heldout = episodes(120)
    explorer_heldout = tuple(
        replace(item, extracted=index >= 30, won=index >= 30)
        for index, item in enumerate(strong_heldout)
    )
    for path, policy, split, checkpoint, base, items in (
        (
            paths["strong_validation_path"],
            "strong",
            "validation",
            strong_checkpoint,
            None,
            strong_validation,
        ),
        (
            paths["validation_path"],
            "explorer",
            "validation",
            explorer_checkpoint,
            strong_sha256,
            tuple(explorer_validation),
        ),
        (
            paths["strong_heldout_path"],
            "strong",
            "heldout",
            strong_checkpoint,
            None,
            strong_heldout,
        ),
        (
            paths["heldout_path"],
            "explorer",
            "heldout",
            explorer_checkpoint,
            strong_sha256,
            explorer_heldout,
        ),
    ):
        _write_report(
            path,
            policy=policy,
            split=split,
            checkpoint=checkpoint,
            base_sha256=base,
            episode_items=items,
        )
    return {
        "checkpoint": explorer_checkpoint,
        "policy": "explorer",
        **paths,
    }


def test_validation_demonstration_discloses_heldout_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence(tmp_path)
    monkeypatch.setattr(
        "botcolosseo.evaluation.extraction_gates._paired_bootstrap_interval",
        lambda values: (sum(values) / len(values), -0.01, 0.5),
    )

    result = build_validation_demonstration(root=tmp_path, **evidence)

    assert result["evidence_tier"] == "validation_demonstration"
    assert result["product_demo_eligible"] is True
    assert result["research_gate_passed"] is False
    assert result["official_test_eligible"] is False
    assert result["validation_failed_checks"] == ["style_ci_lower"]
    assert "heldout_extraction_delta" in result["heldout_failed_checks"]
    assert result["test_cases_accessed"] is False


def test_validation_demonstration_rejects_capability_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence(tmp_path)
    validation_path = Path(evidence["validation_path"])
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    for episode in payload["metrics"]["episodes"]:
        episode["extracted"] = False
        episode["won"] = False
    validation_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "botcolosseo.evaluation.extraction_gates._paired_bootstrap_interval",
        lambda values: (sum(values) / len(values), -0.01, 0.5),
    )

    with pytest.raises(ValueError, match="capability"):
        build_validation_demonstration(root=tmp_path, **evidence)


def test_representative_case_demonstration_discloses_aggregate_failure(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)
    validation_path = Path(evidence["validation_path"])
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    for index, episode in enumerate(payload["metrics"]["episodes"]):
        episode["decisions"] = 300 if index == 0 else episode["decisions"]
        episode["meaningful_loot_regions"] = 4 if index == 0 else 0
        episode["backpack_upgrades"] = 1 if index == 0 else 0
        episode["upgrade_to_extraction_conversions"] = 1 if index == 0 else 0
    validation_path.write_text(json.dumps(payload), encoding="utf-8")

    result = build_validation_demonstration(root=tmp_path, **evidence)

    assert result["evidence_tier"] == "representative_case_demonstration"
    assert result["claim_scope"] == "representative_validation_cases_only"
    assert result["aggregate_style_gate_passed"] is False
    assert result["representative_case_count"] == 1
    assert result["product_demo_eligible"] is True


def test_defensive_case_study_allows_bounded_timeout_regression(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)
    evidence["policy"] = "defensive"
    for key in ("validation_path", "heldout_path"):
        path = Path(evidence[key])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["policy"] = "defensive"
        payload["disengagement_metric_version"] = 3
        path.write_text(json.dumps(payload), encoding="utf-8")

    validation_path = Path(evidence["validation_path"])
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    for index, episode in enumerate(payload["metrics"]["episodes"]):
        episode["decisions"] = 300 if index == 0 else episode["decisions"]
        episode["successful_disengagements"] = 1 if index == 0 else 0
        episode["disengagement_opportunities"] = 1
        episode["meaningful_extractions"] = 1
        episode["timeout_with_value"] = int(index < 6)
    validation_path.write_text(json.dumps(payload), encoding="utf-8")

    result = build_validation_demonstration(root=tmp_path, **evidence)

    assert result["evidence_tier"] == "representative_case_demonstration"
    assert "anti_hack_no_timeout_value_loss" in result["validation_failed_checks"]
    assert result["representative_case_count"] == 1


def test_defensive_case_study_rejects_excessive_timeout_regression(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)
    evidence["policy"] = "defensive"
    for key in ("validation_path", "heldout_path"):
        path = Path(evidence[key])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["policy"] = "defensive"
        payload["disengagement_metric_version"] = 3
        path.write_text(json.dumps(payload), encoding="utf-8")

    validation_path = Path(evidence["validation_path"])
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    for index, episode in enumerate(payload["metrics"]["episodes"]):
        episode["decisions"] = 300 if index == 0 else episode["decisions"]
        episode["successful_disengagements"] = 1 if index == 0 else 0
        episode["disengagement_opportunities"] = 1
        episode["meaningful_extractions"] = 1
        episode["timeout_with_value"] = int(index < 13)
    validation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="maximums"):
        build_validation_demonstration(root=tmp_path, **evidence)
