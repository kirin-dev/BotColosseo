from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from botcolosseo.agents.hybrid_config import load_hybrid_policy_config
from botcolosseo.agents.style_governor import (
    DefensiveGovernorConfig,
    ExplorerGovernorConfig,
)


def test_frozen_hybrid_configs_bind_strong_base_and_public_validation_only() -> None:
    root = Path(__file__).resolve().parents[2]

    defensive = load_hybrid_policy_config(
        root / "configs/m5/hybrid/defensive.yaml",
        root=root,
    )
    explorers = [
        load_hybrid_policy_config(root / path, root=root)
        for path in (
            "configs/m5/hybrid/explorer.yaml",
            "configs/m5/hybrid/explorer_b.yaml",
            "configs/m5/hybrid/explorer_c.yaml",
        )
    ]

    assert defensive.style == "defensive"
    assert all(explorer.style == "explorer" for explorer in explorers)
    assert isinstance(defensive.governor, DefensiveGovernorConfig)
    assert all(
        isinstance(explorer.governor, ExplorerGovernorConfig)
        for explorer in explorers
    )
    assert all(defensive.base_checkpoint == explorer.base_checkpoint for explorer in explorers)
    assert all(
        defensive.base_checkpoint_sha256 == explorer.base_checkpoint_sha256
        for explorer in explorers
    )
    assert all(defensive.scenario_hash == explorer.scenario_hash for explorer in explorers)
    assert defensive.test_cases_accessed is False
    assert all(explorer.test_cases_accessed is False for explorer in explorers)
    assert len(defensive.config_sha256) == 64
    assert all(len(explorer.config_sha256) == 64 for explorer in explorers)


def test_hybrid_config_rejects_unknown_fields_and_test_access(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    source = yaml.safe_load(
        (root / "configs/m5/hybrid/defensive.yaml").read_text(encoding="utf-8")
    )
    source["unknown"] = True
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        load_hybrid_policy_config(path, root=root)

    source.pop("unknown")
    source["test_cases_accessed"] = True
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ValueError, match="must not access"):
        load_hybrid_policy_config(path, root=root)


def test_hybrid_config_rejects_style_schema_mismatch(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    source = yaml.safe_load(
        (root / "configs/m5/hybrid/defensive.yaml").read_text(encoding="utf-8")
    )
    source["style"] = "explorer"
    path = tmp_path / "mismatch.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")

    with pytest.raises(ValueError, match="Explorer governor fields"):
        load_hybrid_policy_config(path, root=root)
