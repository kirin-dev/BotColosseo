from __future__ import annotations

import json
from pathlib import Path

import pytest

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.evaluation.m6_release import M6_POLICIES
from botcolosseo.evaluation.m6_showcase_config import (
    M6_POLICY_INPUTS,
    build_m6_showcase_config,
    write_m6_showcase_config,
)


def _setup(root: Path) -> Path:
    hashes = {}
    for index, policy_id in enumerate(M6_POLICIES):
        checkpoint = root / M6_POLICY_INPUTS[policy_id][1]
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
        hashes[policy_id] = sha256_file(checkpoint)
    metrics = root / "reports/m6/showcase-metrics.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "stage": "m6",
                "split": "validation",
                "passed": True,
                "style_gate_passed": True,
                "retention_gate_passed": True,
                "difficulty_gate_passed": True,
                "episodes": 1800,
                "checkpoint_sha256": hashes,
                "headline_cards": [
                    {"label": f"Metric {index}", "value": str(index)}
                    for index in range(6)
                ],
                "case_contrast_scores": {"case": 1.0},
                "decision_contrast_scores": {"case": [0.0, 1.0]},
                "upstream_sha256": {
                    "m4": "1" * 64,
                    "defensive": "2" * 64,
                    "explorer": "3" * 64,
                    "difficulty": "4" * 64,
                },
                "test_cases_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    return metrics


def test_m6_config_is_derived_from_real_checkpoint_hashes(tmp_path: Path) -> None:
    metrics = _setup(tmp_path)

    config = build_m6_showcase_config(root=tmp_path, metrics_path=metrics)

    assert config["stage"] == "m6"
    assert [row["id"] for row in config["policies"]] == list(M6_POLICIES)
    assert config["metrics"] == "reports/m6/showcase-metrics.json"
    assert config["evidence_dir"] == "reports/showcase/m6"


def test_m6_config_rejects_checkpoint_drift(tmp_path: Path) -> None:
    metrics = _setup(tmp_path)
    checkpoint = tmp_path / M6_POLICY_INPUTS["defensive"][1]
    checkpoint.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="defensive"):
        build_m6_showcase_config(root=tmp_path, metrics_path=metrics)


def test_m6_config_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "m6.yaml"
    write_m6_showcase_config({"stage": "m6"}, output)

    with pytest.raises(FileExistsError, match="overwrite"):
        write_m6_showcase_config({"stage": "m6"}, output)
