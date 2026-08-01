from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from botcolosseo.cli.admit_extraction_showcase import (
    build_directional_showcase_admission,
)
from botcolosseo.cli.check_extraction_aggressive_prerequisite import (
    check_aggressive_prerequisite,
)
from botcolosseo.data.demonstrations import sha256_file
from tests.unit.test_extraction_gates import episodes


def _write_report(
    path: Path,
    *,
    policy: str,
    split: str,
    checkpoint: Path,
    base_sha256: str | None,
    episode_items: tuple[object, ...],
) -> None:
    payload = {
        "policy": policy,
        "metric_schema_version": 2,
        "split": split,
        "complete": True,
        "test_cases_accessed": False,
        "actor_privilege_violations": 0,
        "fair_actor_observation_only": True,
        "checkpoint": str(checkpoint.relative_to(path.parent)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "base_checkpoint_sha256": base_sha256,
        "protocol_sha256": "p" * 64,
        "scenario_hash": "s" * 64,
        "metrics": {"episodes": [asdict(item) for item in episode_items]},
    }
    if policy == "defensive":
        payload["disengagement_metric_version"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")


def _evidence(tmp_path: Path) -> dict[str, Path]:
    strong_checkpoint = tmp_path / "strong.pt"
    aggressive_checkpoint = tmp_path / "aggressive.pt"
    strong_checkpoint.write_bytes(b"strong")
    aggressive_checkpoint.write_bytes(b"aggressive")
    strong_sha256 = sha256_file(strong_checkpoint)

    strong_validation = tmp_path / "strong-validation.json"
    aggressive_validation = tmp_path / "aggressive-validation.json"
    strong_heldout = tmp_path / "strong-heldout.json"
    aggressive_heldout = tmp_path / "aggressive-heldout.json"
    strong_validation_episodes = episodes(240)
    aggressive_validation_episodes = list(strong_validation_episodes)
    for index in range(140):
        aggressive_validation_episodes[index] = replace(
            aggressive_validation_episodes[index],
            valid_hits=3,
            kills=1,
            favorable_encounter_initiations=1,
            kill_to_cache_conversions=1,
            cache_to_extraction_conversions=1,
            aggressive_chains=1,
        )
    for index in range(140, 240):
        aggressive_validation_episodes[index] = replace(
            aggressive_validation_episodes[index],
            valid_hits=1,
        )
    _write_report(
        strong_validation,
        policy="strong",
        split="validation",
        checkpoint=strong_checkpoint,
        base_sha256=None,
        episode_items=strong_validation_episodes,
    )
    _write_report(
        aggressive_validation,
        policy="aggressive",
        split="validation",
        checkpoint=aggressive_checkpoint,
        base_sha256=strong_sha256,
        episode_items=tuple(aggressive_validation_episodes),
    )
    strong_heldout_episodes = list(episodes(120))
    aggressive_heldout_episodes = list(episodes(120))
    strong_wins = {
        "aggressive": 22,
        "defensive": 15,
        "explorer": 7,
        "strong": 25,
    }
    aggressive_wins = {
        "aggressive": 22,
        "defensive": 11,
        "explorer": 8,
        "strong": 26,
    }
    seen = {opponent: 0 for opponent in strong_wins}
    for index, item in enumerate(strong_heldout_episodes):
        opponent = item.opponent_style
        strong_heldout_episodes[index] = replace(
            item,
            won=seen[opponent] < strong_wins[opponent],
        )
        aggressive_heldout_episodes[index] = replace(
            aggressive_heldout_episodes[index],
            won=seen[opponent] < aggressive_wins[opponent],
        )
        seen[opponent] += 1
    _write_report(
        strong_heldout,
        policy="strong",
        split="heldout",
        checkpoint=strong_checkpoint,
        base_sha256=None,
        episode_items=tuple(strong_heldout_episodes),
    )
    _write_report(
        aggressive_heldout,
        policy="aggressive",
        split="heldout",
        checkpoint=aggressive_checkpoint,
        base_sha256=strong_sha256,
        episode_items=tuple(aggressive_heldout_episodes),
    )
    return {
        "checkpoint": aggressive_checkpoint,
        "validation_path": aggressive_validation,
        "strong_validation_path": strong_validation,
        "heldout_path": aggressive_heldout,
        "strong_heldout_path": strong_heldout,
    }


def _explorer_evidence(tmp_path: Path) -> dict[str, Path | str]:
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
    strong_validation_episodes = episodes(240)
    explorer_validation_episodes = list(strong_validation_episodes)
    for index in range(140):
        explorer_validation_episodes[index] = replace(
            explorer_validation_episodes[index],
            meaningful_loot_regions=3,
            backpack_upgrades=1,
            upgrade_to_extraction_conversions=1,
        )
    for index in range(140, 240):
        explorer_validation_episodes[index] = replace(
            explorer_validation_episodes[index],
            meaningful_loot_regions=0,
        )
    _write_report(
        paths["strong_validation_path"],
        policy="strong",
        split="validation",
        checkpoint=strong_checkpoint,
        base_sha256=None,
        episode_items=strong_validation_episodes,
    )
    _write_report(
        paths["validation_path"],
        policy="explorer",
        split="validation",
        checkpoint=explorer_checkpoint,
        base_sha256=strong_sha256,
        episode_items=tuple(explorer_validation_episodes),
    )
    _write_report(
        paths["strong_heldout_path"],
        policy="strong",
        split="heldout",
        checkpoint=strong_checkpoint,
        base_sha256=None,
        episode_items=episodes(120),
    )
    _write_report(
        paths["heldout_path"],
        policy="explorer",
        split="heldout",
        checkpoint=explorer_checkpoint,
        base_sha256=strong_sha256,
        episode_items=episodes(120),
    )
    return {
        "checkpoint": explorer_checkpoint,
        "policy": "explorer",
        **paths,
    }


def test_directional_admission_accepts_only_ci_lower_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence(tmp_path)
    monkeypatch.setattr(
        "botcolosseo.evaluation.extraction_gates._paired_bootstrap_interval",
        lambda values: (sum(values) / len(values), -0.01, 0.5),
    )

    result = build_directional_showcase_admission(root=tmp_path, **evidence)

    assert result["research_failed_checks"] == [
        "style_ci_lower",
        "heldout_worst_opponent_retention",
    ]
    assert result["research_validation_failed_checks"] == ["style_ci_lower"]
    assert result["original_heldout_gate_passed"] is False
    assert result["original_heldout_failed_checks"] == [
        "heldout_worst_opponent_retention"
    ]
    assert result["research_gate_passed"] is False
    assert result["showcase_eligible"] is True
    assert all(check["passed"] for check in result["showcase_heldout_checks"])
    assert result["per_opponent_heldout"]["explorer"][
        "styled_win_rate"
    ] == pytest.approx(8 / 30)
    assert result["direction_counts"]["positive_pairs"] == 140
    assert result["direction_counts"]["negative_pairs"] == 100


def test_explorer_admission_uses_rule_frozen_before_heldout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _explorer_evidence(tmp_path)
    monkeypatch.setattr(
        "botcolosseo.evaluation.extraction_gates._paired_bootstrap_interval",
        lambda values: (sum(values) / len(values), -0.01, 0.5),
    )

    result = build_directional_showcase_admission(root=tmp_path, **evidence)

    assert result["policy"] == "explorer"
    assert result["admission_rule_timing"] == "pre_heldout_product_rule"
    assert result["research_failed_checks"] == ["style_ci_lower"]
    assert result["original_heldout_gate_passed"] is True
    assert result["direction_counts"] == {
        "showcase_chain_kind": "upgrade_to_extraction",
        "positive_pairs": 140,
        "negative_pairs": 100,
        "unchanged_pairs": 0,
        "new_showcase_chains": 140,
        "lost_showcase_chains": 0,
    }


def test_directional_admission_rejects_non_ci_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence(tmp_path)
    validation_path = evidence["validation_path"]
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    for episode in payload["metrics"]["episodes"]:
        episode["extracted"] = False
    validation_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "botcolosseo.evaluation.extraction_gates._paired_bootstrap_interval",
        lambda values: (sum(values) / len(values), -0.01, 0.5),
    )

    with pytest.raises(ValueError, match="sole research failure"):
        build_directional_showcase_admission(root=tmp_path, **evidence)


def test_prerequisite_accepts_bound_admission_and_rejects_drift(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "aggressive"
    directory.mkdir()
    evidence = tmp_path / "validation.json"
    evidence.write_bytes(b"evidence")
    showcase = directory / "showcase.pt"
    showcase.write_bytes(b"policy")
    admission = {
        "admission_schema_version": 1,
        "admission_kind": "directional_showcase",
        "admission_rule_timing": "post_heldout_product_review",
        "policy": "aggressive",
        "showcase_eligible": True,
        "research_gate_passed": False,
        "research_failed_checks": [
            "style_ci_lower",
            "heldout_worst_opponent_retention",
        ],
        "research_validation_failed_checks": ["style_ci_lower"],
        "original_heldout_gate_passed": False,
        "original_heldout_failed_checks": [
            "heldout_worst_opponent_retention"
        ],
        "showcase_heldout_checks": [{"name": "relative", "passed": True}],
        "actor_privilege_violations": 0,
        "test_cases_accessed": False,
        "showcase_checkpoint_sha256": sha256_file(showcase),
        "evidence_sha256": {
            "validation.json": sha256_file(evidence),
        },
    }
    (directory / "showcase-admission.json").write_text(
        json.dumps(admission), encoding="utf-8"
    )
    assert check_aggressive_prerequisite(
        tmp_path, Path("aggressive")
    ) == "directional_showcase"

    evidence.write_bytes(b"drift")
    with pytest.raises(ValueError, match="evidence drifted"):
        check_aggressive_prerequisite(tmp_path, Path("aggressive"))


def test_prerequisite_prefers_valid_research_selection(tmp_path: Path) -> None:
    directory = tmp_path / "aggressive"
    directory.mkdir()
    selected = directory / "selected.pt"
    selected.write_bytes(b"selected")
    (directory / "selection.json").write_text(
        json.dumps(
            {
                "gate_schema_version": 2,
                "policy": "aggressive",
                "eligible": True,
                "test_cases_accessed": False,
                "selected_checkpoint_sha256": sha256_file(selected),
            }
        ),
        encoding="utf-8",
    )

    assert check_aggressive_prerequisite(
        tmp_path, Path("aggressive")
    ) == "research_selection"
