from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from botcolosseo.evaluation.synthetic_user_study import (
    generate_synthetic_user_study,
)
from botcolosseo.evaluation.user_study import (
    STYLES,
    analyze_user_study,
    prepare_user_study,
)


def test_synthetic_user_study_is_complete_and_explicit(
    tmp_path: Path,
) -> None:
    clips = {}
    for style_index, style in enumerate(STYLES):
        paths = (tmp_path / f"{style}-1.mp4", tmp_path / f"{style}-2.mp4")
        for variant, path in enumerate(paths):
            path.write_bytes(b"video" + bytes([style_index, variant]))
        clips[style] = paths
    package = tmp_path / "package"
    prepare_user_study(clips, output_dir=package, assignment_count=10)
    responses = tmp_path / "responses.csv"
    provenance_path = tmp_path / "provenance.json"
    config = (
        Path(__file__).resolve().parents[2]
        / "configs/m6/synthetic-user-study.json"
    )

    provenance = generate_synthetic_user_study(
        package,
        config,
        responses_path=responses,
        provenance_path=provenance_path,
    )
    with responses.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = analyze_user_study(
        package,
        responses,
        synthetic_data=True,
    )

    assert len(rows) == 60
    assert len({row["respondent_id"] for row in rows}) == 10
    assert provenance["synthetic_data"] is True
    assert provenance["human_participants"] is False
    assert not Path(provenance["config_path"]).is_absolute()
    assert json.loads(provenance_path.read_text(encoding="utf-8")) == provenance
    assert summary["macro_recognition_rate"] == pytest.approx(0.85)
    assert summary["per_style"]["aggressive"]["recognition_rate"] == 0.9
    assert summary["per_style"]["defensive"]["recognition_rate"] == 0.8
    assert summary["per_style"]["explorer"]["recognition_rate"] == 0.85
