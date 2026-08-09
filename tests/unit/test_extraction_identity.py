from __future__ import annotations

import json
from pathlib import Path

import pytest

from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.evaluation.extraction_identity import (
    IDENTITY_COMPONENTS,
    build_experiment_identity,
    validate_experiment_identity,
)


def _root(tmp_path: Path) -> Path:
    for name, relative in IDENTITY_COMPONENTS.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    wad = tmp_path / IDENTITY_COMPONENTS["scenario_wad"]
    manifest = tmp_path / IDENTITY_COMPONENTS["scenario_manifest"]
    manifest.write_text(
        json.dumps({"wad_sha256": sha256_file(wad)}),
        encoding="utf-8",
    )
    return tmp_path


def test_identity_binds_all_experiment_components(tmp_path: Path) -> None:
    root = _root(tmp_path)
    identity = build_experiment_identity(root=root, git_commit="a" * 40)

    validate_experiment_identity(root=root, payload=identity)
    assert identity["scenario_hash"] == identity["components"]["scenario_wad"]["sha256"]
    assert set(identity["components"]) == set(IDENTITY_COMPONENTS)


@pytest.mark.parametrize(
    "component",
    ("scenario_config", "layout_generator", "game_rules", "teacher",
     "evaluation_protocol", "metric_implementation", "gate_implementation"),
)
def test_identity_detects_component_drift(tmp_path: Path, component: str) -> None:
    root = _root(tmp_path)
    identity = build_experiment_identity(root=root, git_commit="b" * 40)
    (root / IDENTITY_COMPONENTS[component]).write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="components drifted"):
        validate_experiment_identity(root=root, payload=identity)


def test_identity_changes_with_git_commit(tmp_path: Path) -> None:
    root = _root(tmp_path)

    first = build_experiment_identity(root=root, git_commit="c" * 40)
    second = build_experiment_identity(root=root, git_commit="d" * 40)

    assert first["experiment_identity_sha256"] != second[
        "experiment_identity_sha256"
    ]
