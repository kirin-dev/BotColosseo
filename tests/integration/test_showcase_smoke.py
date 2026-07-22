from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

import botcolosseo.cli.render_showcase as showcase_cli
from botcolosseo.evaluation.showcase import load_showcase_config


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("BOTCOLOSSEO_RUN_SHOWCASE_SMOKE") != "1",
    reason="set BOTCOLOSSEO_RUN_SHOWCASE_SMOKE=1 for real ViZDoom media",
)
def test_real_m2_showcase_writes_non_public_hash_bound_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path.cwd()
    checkpoint_root = Path(os.environ["BOTCOLOSSEO_M2_ARTIFACT_ROOT"])
    config = load_showcase_config(
        Path("configs/showcase/development.yaml"), root=root
    )
    isolated = replace(
        config,
        output_dir=tmp_path / "media",
        evidence_dir=tmp_path / "evidence",
    )
    monkeypatch.setattr(
        showcase_cli, "load_showcase_config", lambda path, root: isolated
    )

    manifest = showcase_cli.render_showcase(
        root=root,
        config_path=config.config_path,
        checkpoint_root=checkpoint_root,
        device="cuda:0",
    )

    assert manifest["publication"] is False
    assert manifest["split"] == "validation"
    assert manifest["official_test_result"] is False
    assert manifest["test_cases_accessed"] is False
    assert manifest["gate_passed"] is False
    assert (tmp_path / "media/development-ppo.mp4").stat().st_size > 10_000
    assert (tmp_path / "media/development-bc.mp4").stat().st_size > 10_000
    assert (tmp_path / "media/development-comparison.gif").stat().st_size > 10_000
    saved = json.loads((tmp_path / "evidence/manifest.json").read_text())
    assert saved["run_identity"] == manifest["run_identity"]
