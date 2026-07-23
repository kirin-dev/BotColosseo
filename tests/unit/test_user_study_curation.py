from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.envs.video import write_mp4
from botcolosseo.evaluation.user_study import STYLES
from botcolosseo.evaluation.user_study_curation import (
    curate_user_study_clips,
)


def _fixture(root: Path) -> Path:
    candidate_dir = root / "artifacts/candidates"
    candidate_dir.mkdir(parents=True)
    rows = []
    selections = []
    for style in STYLES:
        for variant in (1, 2):
            filename = f"{style}-{variant}.mp4"
            video = candidate_dir / filename
            frames = [
                np.full((8, 8, 3), index, dtype=np.uint8)
                for index in range(25)
            ]
            write_mp4(frames, video, fps=1)
            digest = sha256_file(video)
            rows.append(
                {
                    "style": style,
                    "rank": variant,
                    "case_id": f"case:{style}:{variant}",
                    "path": filename,
                    "sha256": digest,
                    "frame_count": 25,
                    "fps": 1,
                    "replay": {
                        "protocol_inconsistent": False,
                        "observer_steps": [
                            {
                                "self_health": 100 - index,
                                "opponent_health": 100,
                                "events": ["HIT"] if index == 12 else [],
                            }
                            for index in range(25)
                        ],
                    },
                }
            )
            selections.append(
                {
                    "style": style,
                    "variant": variant,
                    "source_manifest": "artifacts/candidates/manifest.json",
                    "source_rank": variant,
                    "case_id": f"case:{style}:{variant}",
                    "source_path": filename,
                    "source_sha256": digest,
                    "start_frame": 0,
                    "end_frame": 25,
                    "reason": "test fixture",
                }
            )
    manifest = {
        "stage": "m6-user-study-candidates",
        "test_cases_accessed": False,
        "candidates": rows,
    }
    manifest_path = candidate_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = sha256_file(manifest_path)
    for selection in selections:
        selection["source_manifest_sha256"] = manifest_sha
    config = root / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection": "curated validation showcase",
                "clips": selections,
            }
        ),
        encoding="utf-8",
    )
    return config


def test_curate_user_study_clips_binds_sources_and_windows(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)

    result = curate_user_study_clips(
        config,
        output_dir=Path("output"),
        root=tmp_path,
    )

    assert result["clip_count"] == 6
    assert result["test_cases_accessed"] is False
    assert all(row["duration_seconds"] == 25 for row in result["clips"])
    assert all(row["visible_summary"]["events"] == {"HIT": 1} for row in result["clips"])


def test_curate_user_study_clips_rejects_tampered_source(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    (tmp_path / "artifacts/candidates/aggressive-1.mp4").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="video hash"):
        curate_user_study_clips(
            config,
            output_dir=Path("output"),
            root=tmp_path,
        )
