from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from botcolosseo.evaluation.user_study import (
    RESPONSE_FIELDS,
    STYLES,
    analyze_user_study,
    prepare_user_study,
    render_user_study_chart,
)


def _clips(root: Path) -> dict[str, tuple[Path, Path]]:
    result = {}
    for index, style in enumerate(STYLES):
        paths = (root / f"{style}-1.mp4", root / f"{style}-2.mp4")
        for variant, path in enumerate(paths):
            path.write_bytes(b"video" + bytes([index, variant]))
        result[style] = paths
    return result


def _responses(package: Path, output: Path) -> None:
    assignments: dict[str, list[str]] = {}
    with (package / "assignments.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            assignments.setdefault(row["assignment_id"], []).append(row["clip_id"])
    key = json.loads((package / "answer-key.json").read_text(encoding="utf-8"))
    true_styles = {row["clip_id"]: row["true_style"] for row in key["clips"]}
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESPONSE_FIELDS)
        writer.writeheader()
        for respondent_index, assignment in enumerate(sorted(assignments)[:2]):
            for clip_id in assignments[assignment]:
                writer.writerow(
                    {
                        "respondent_id": f"r{respondent_index + 1}",
                        "assignment_id": assignment,
                        "clip_id": clip_id,
                        "style_choice": true_styles[clip_id],
                        "style_clarity": 5,
                        "perceived_difficulty": 3,
                    }
                )


def test_prepare_user_study_is_deterministic_and_counterbalanced(
    tmp_path: Path,
) -> None:
    clips = _clips(tmp_path)

    first = prepare_user_study(
        clips, output_dir=tmp_path / "first", assignment_count=10, seed=7
    )
    prepare_user_study(
        clips, output_dir=tmp_path / "second", assignment_count=10, seed=7
    )

    assert first["assignment_count"] == 10
    assert first["clips_per_style"] == 2
    assert first["clip_count"] == 6
    assert first["styles"] == list(STYLES)
    assert first["test_cases_accessed"] is False
    assert (tmp_path / "first/response-template.csv").is_file()
    instructions = (tmp_path / "first/participant-instructions.md").read_text(
        encoding="utf-8"
    )
    assert "第一视角 Bot" in instructions
    assert "OPP HP" in instructions
    assert (
        (tmp_path / "first/assignments.csv").read_text(encoding="utf-8")
        == (tmp_path / "second/assignments.csv").read_text(encoding="utf-8")
    )
    first_key = json.loads(
        (tmp_path / "first/answer-key.json").read_text(encoding="utf-8")
    )
    assert {row["true_style"] for row in first_key["clips"]} == set(STYLES)
    assert len(first_key["clips"]) == 6
    assert all(
        style not in row["clip_id"]
        for row in first_key["clips"]
        for style in STYLES
    )


def test_prepare_user_study_refuses_overwrite(tmp_path: Path) -> None:
    clips = _clips(tmp_path)
    output = tmp_path / "package"
    prepare_user_study(clips, output_dir=output)

    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_user_study(clips, output_dir=output)


def test_analysis_reports_recognition_and_hashes(tmp_path: Path) -> None:
    package = tmp_path / "package"
    prepare_user_study(_clips(tmp_path), output_dir=package, assignment_count=2)
    responses = tmp_path / "responses.csv"
    _responses(package, responses)

    result = analyze_user_study(package, responses)

    assert result["respondents"] == 2
    assert result["responses"] == 12
    assert result["macro_recognition_rate"] == 1
    assert result["micro_recognition_rate"] == 1
    assert result["small_sample_product_study"] is True
    assert result["synthetic_data"] is False
    assert result["human_participants"] is True
    assert all(
        result["per_style"][style]["recognition_rate"] == 1 for style in STYLES
    )

    chart = render_user_study_chart(result, tmp_path / "user-study.png")
    assert chart.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert chart.stat().st_size > 10_000


def test_analysis_marks_synthetic_preflight(tmp_path: Path) -> None:
    package = tmp_path / "package"
    prepare_user_study(_clips(tmp_path), output_dir=package, assignment_count=2)
    responses = tmp_path / "responses.csv"
    _responses(package, responses)

    result = analyze_user_study(
        package,
        responses,
        synthetic_data=True,
    )

    assert result["synthetic_data"] is True
    assert result["human_participants"] is False
    chart = render_user_study_chart(result, tmp_path / "synthetic.png")
    assert chart.stat().st_size > 10_000


def test_analysis_rejects_incomplete_respondent(tmp_path: Path) -> None:
    package = tmp_path / "package"
    prepare_user_study(_clips(tmp_path), output_dir=package, assignment_count=1)
    responses = tmp_path / "responses.csv"
    _responses(package, responses)
    rows = responses.read_text(encoding="utf-8").splitlines()
    responses.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete"):
        analyze_user_study(package, responses)


def test_analysis_rejects_tampered_clip(tmp_path: Path) -> None:
    package = tmp_path / "package"
    prepare_user_study(_clips(tmp_path), output_dir=package, assignment_count=1)
    responses = tmp_path / "responses.csv"
    _responses(package, responses)
    clip = next((package / "clips").iterdir())
    clip.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="hash"):
        analyze_user_study(package, responses)
