from __future__ import annotations

import json
from pathlib import Path

import pytest

from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.evaluation.extraction_release import audit_extraction_release


def test_extraction_release_audit_binds_artifact_hashes(tmp_path: Path) -> None:
    artifact = tmp_path / "clip.mp4"
    artifact.write_bytes(b"real-validation-replay")
    report = tmp_path / "manifest.json"
    report.write_text(
        json.dumps(
            {
                "acceptance": {"all_clips_validation_only": True},
                "artifacts": [
                    {
                        "bytes": artifact.stat().st_size,
                        "path": "clip.mp4",
                        "sha256": sha256_file(artifact),
                    }
                ],
                "stage": "crystal-run-extraction-v2-showcase",
                "test_cases_accessed": False,
            }
        ),
        encoding="utf-8",
    )

    assert audit_extraction_release(report, root=tmp_path)["test_cases_accessed"] is False

    artifact.write_bytes(b"drift")
    with pytest.raises(ValueError, match="size drift"):
        audit_extraction_release(report, root=tmp_path)
