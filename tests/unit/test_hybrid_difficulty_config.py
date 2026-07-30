from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from botcolosseo.evaluation.hybrid_difficulty_config import (
    load_hybrid_difficulty_product_config,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(root: Path) -> Path:
    scenario_hash = "9" * 64
    paths = {
        "difficulty": root / "configs/difficulty.yaml",
        "cases": root / "configs/m2/validation.json",
        "scenario": root / "assets/scenarios/crystal_run/manifest.json",
        "base": root / "reports/m5/difficulty/formal/manifest.json",
        "defensive_config": root / "configs/m5/hybrid/defensive.yaml",
        "defensive_manifest": (
            root / "reports/m5/hybrid/defensive/formal-a/manifest.json"
        ),
        "explorer_config": root / "configs/m5/hybrid/explorer_c.yaml",
        "explorer_manifest": (
            root / "reports/m5/hybrid/explorer/formal-c/manifest.json"
        ),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["difficulty"].write_text("easy: {}\nnormal: {}\n", encoding="utf-8")
    paths["cases"].write_text("[]\n", encoding="utf-8")
    paths["scenario"].write_text(
        yaml.safe_dump({"wad_sha256": scenario_hash}),
        encoding="utf-8",
    )
    for key in (
        "base",
        "defensive_manifest",
        "explorer_manifest",
    ):
        paths[key].write_text(json.dumps({"fixture": key}), encoding="utf-8")
    paths["defensive_config"].write_text("style: defensive\n", encoding="utf-8")
    paths["explorer_config"].write_text("style: explorer\n", encoding="utf-8")

    product = root / "configs/m5/hybrid/difficulty-product.yaml"
    product.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "stage": "m5_hybrid_all_style_difficulty",
                "test_cases_accessed": False,
                "difficulty": {
                    "path": "configs/difficulty.yaml",
                    "expected_sha256": _sha256(paths["difficulty"]),
                },
                "cases": {
                    "path": "configs/m2/validation.json",
                    "expected_sha256": _sha256(paths["cases"]),
                },
                "scenario": {
                    "path": "assets/scenarios/crystal_run/manifest.json",
                    "expected_sha256": _sha256(paths["scenario"]),
                    "wad_sha256": scenario_hash,
                },
                "sources": {
                    "base_aggressive": {
                        "manifest": (
                            "reports/m5/difficulty/formal/manifest.json"
                        ),
                        "expected_sha256": _sha256(paths["base"]),
                    },
                    "defensive": {
                        "governor_config": (
                            "configs/m5/hybrid/defensive.yaml"
                        ),
                        "governor_config_sha256": _sha256(
                            paths["defensive_config"]
                        ),
                        "hard_manifest": (
                            "reports/m5/hybrid/defensive/formal-a/manifest.json"
                        ),
                        "hard_manifest_sha256": _sha256(
                            paths["defensive_manifest"]
                        ),
                    },
                    "explorer": {
                        "governor_config": (
                            "configs/m5/hybrid/explorer_c.yaml"
                        ),
                        "governor_config_sha256": _sha256(
                            paths["explorer_config"]
                        ),
                        "hard_manifest": (
                            "reports/m5/hybrid/explorer/formal-c/manifest.json"
                        ),
                        "hard_manifest_sha256": _sha256(
                            paths["explorer_manifest"]
                        ),
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return product


def test_frozen_hybrid_difficulty_product_config_binds_sources(
    tmp_path: Path,
) -> None:
    product = _write_fixture(tmp_path)

    config = load_hybrid_difficulty_product_config(
        product.relative_to(tmp_path),
        root=tmp_path,
    )

    assert config.test_cases_accessed is False
    assert config.defensive.style == "defensive"
    assert config.explorer.style == "explorer"
    assert config.defensive.governor_config.name == "defensive.yaml"
    assert config.explorer.governor_config.name == "explorer_c.yaml"
    assert len(config.config_sha256) == 64


def test_hybrid_difficulty_product_config_rejects_hash_drift(
    tmp_path: Path,
) -> None:
    product = _write_fixture(tmp_path)
    payload = yaml.safe_load(product.read_text(encoding="utf-8"))
    payload["difficulty"]["expected_sha256"] = "0" * 64
    path = tmp_path / "drift.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="difficulty config hash drifted"):
        load_hybrid_difficulty_product_config(path, root=tmp_path)
