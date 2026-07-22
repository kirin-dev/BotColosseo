from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from botcolosseo.evaluation.showcase import (
    case_id,
    load_showcase_cases,
    load_showcase_config,
)


def test_development_config_is_non_public_and_hash_bound() -> None:
    config = load_showcase_config(
        Path("configs/showcase/development.yaml"), root=Path.cwd()
    )

    assert config.stage == "development"
    assert config.publication is False
    assert [policy.policy_id for policy in config.policies] == ["ppo", "bc"]
    assert config.metrics_path is None
    assert config.render.gif_max_bytes == 10_000_000
    assert config.output_dir == Path.cwd() / "artifacts/showcase-development/media"


def test_m4_cases_are_validation_only_and_cover_four_scripts() -> None:
    cases = load_showcase_cases(
        Path("configs/showcase/m4-validation.json"),
        root=Path.cwd(),
        expected_count=8,
    )

    assert {case.split for case in cases} == {"validation"}
    assert {case.learner_side for case in cases} == {"host", "opponent"}
    assert {case.opponent for case in cases} == {
        "fixed_route",
        "objective_first",
        "aggressive_script",
        "defensive_script",
    }
    assert len({case_id(case) for case in cases}) == 8


def test_case_manifest_rejects_source_hash_drift(tmp_path: Path) -> None:
    (tmp_path / "validation.json").write_text("[]\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "source": "validation.json",
        "source_sha256": "0" * 64,
        "cases": [],
    }
    path = tmp_path / "showcase.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="source hash"):
        load_showcase_cases(path, root=tmp_path, expected_count=0)


def test_publication_config_rejects_test_split(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "stage": "m4",
        "publication": True,
        "split": "test",
        "cases": "configs/showcase/m4-validation.json",
        "metrics": "reports/m4/validation/summary.json",
        "policies": [
            {
                "id": "strong_base",
                "label": "Strong Base",
                "checkpoint": "runs/m3/selected.pt",
                "expected_sha256": "1" * 64,
            },
            {
                "id": "aggressive",
                "label": "Aggressive",
                "checkpoint": "runs/m4/aggressive/selected.pt",
                "expected_sha256": "2" * 64,
            },
        ],
        "render": {
            "fps": 10,
            "gif_seconds": 18,
            "gif_max_bytes": 10_000_000,
            "max_decisions": 525,
            "output_dir": "docs/assets/showcase",
        },
        "evidence_dir": "reports/showcase/m4",
    }
    path = tmp_path / "m4.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="validation"):
        load_showcase_config(path, root=Path.cwd())
