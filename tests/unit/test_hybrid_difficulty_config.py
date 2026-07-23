from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from botcolosseo.evaluation.hybrid_difficulty_config import (
    load_hybrid_difficulty_product_config,
)


def test_frozen_hybrid_difficulty_product_config_binds_real_sources() -> None:
    root = Path(__file__).resolve().parents[2]

    config = load_hybrid_difficulty_product_config(
        Path("configs/m5/hybrid/difficulty-product.yaml"),
        root=root,
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
    root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load(
        (root / "configs/m5/hybrid/difficulty-product.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["difficulty"]["expected_sha256"] = "0" * 64
    path = tmp_path / "drift.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="difficulty config hash drifted"):
        load_hybrid_difficulty_product_config(path, root=root)
