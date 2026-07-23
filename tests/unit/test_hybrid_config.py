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
    explorer = load_hybrid_policy_config(
        root / "configs/m5/hybrid/explorer.yaml",
        root=root,
    )

    assert defensive.style == "defensive"
    assert explorer.style == "explorer"
    assert isinstance(defensive.governor, DefensiveGovernorConfig)
    assert isinstance(explorer.governor, ExplorerGovernorConfig)
    assert defensive.base_checkpoint == explorer.base_checkpoint
    assert defensive.base_checkpoint_sha256 == explorer.base_checkpoint_sha256
    assert defensive.scenario_hash == explorer.scenario_hash
    assert defensive.test_cases_accessed is explorer.test_cases_accessed is False
    assert len(defensive.config_sha256) == len(explorer.config_sha256) == 64


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
