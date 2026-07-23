from __future__ import annotations

import json
from pathlib import Path

import pytest

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.evaluation.project_closeout import (
    audit_project_closeout,
    build_project_closeout,
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixture(root: Path) -> tuple[dict[str, Path], dict[str, object]]:
    responses = root / "responses.csv"
    responses.write_text("synthetic\n", encoding="utf-8")
    study = _write_json(
        root / "study.json",
        {
            "stage": "m6-user-study-package",
            "clip_count": 6,
            "clips_per_style": 2,
            "assignment_count": 10,
            "test_cases_accessed": False,
        },
    )
    summary = _write_json(
        root / "summary.json",
        {
            "stage": "m6-user-study-analysis",
            "synthetic_data": True,
            "human_participants": False,
            "respondents": 10,
            "responses": 60,
            "macro_recognition_rate": 0.85,
            "micro_recognition_rate": 0.85,
            "responses_sha256": sha256_file(responses),
            "package_manifest_sha256": sha256_file(study),
            "per_style": {
                style: {"recognition_rate": rate}
                for style, rate in (
                    ("aggressive", 0.9),
                    ("defensive", 0.8),
                    ("explorer", 0.85),
                )
            },
            "test_cases_accessed": False,
        },
    )
    provenance = _write_json(
        root / "provenance.json",
        {
            "stage": "m6-synthetic-user-study-preflight",
            "synthetic_data": True,
            "human_participants": False,
            "respondent_count": 10,
            "response_count": 60,
            "responses_sha256": sha256_file(responses),
            "test_cases_accessed": False,
        },
    )
    curation = _write_json(
        root / "curation.json",
        {
            "stage": "m6-curated-validation-clips",
            "clip_count": 6,
            "clips_per_style": 2,
            "test_cases_accessed": False,
        },
    )
    policy_archive = root / "policy.tar.gz"
    policy_archive.write_bytes(b"policy")
    release = _write_json(
        root / "release.json",
        {
            "stage": "hybrid-product-release",
            "audit_passed": True,
            "archive_sha256": sha256_file(policy_archive),
            "test_cases_accessed": False,
        },
    )
    paths = {
        "synthetic_summary_path": summary,
        "synthetic_provenance_path": provenance,
        "synthetic_responses_path": responses,
        "curation_manifest_path": curation,
        "study_manifest_path": study,
        "release_record_path": release,
        "policy_archive_path": policy_archive,
    }
    for name in (
        "study_archive_path",
        "curated_archive_path",
        "future_plan_path",
        "readme_path",
        "readme_cn_path",
    ):
        path = root / name
        path.write_text(name, encoding="utf-8")
        paths[name] = path
    return paths, json.loads(summary.read_text(encoding="utf-8"))


def test_project_closeout_builds_and_detects_drift(tmp_path: Path) -> None:
    paths, summary = _fixture(tmp_path)

    payload = build_project_closeout(root=tmp_path, **paths)

    assert payload["current_scenario_development_complete"] is True
    assert payload["human_participants"] is False
    assert payload["recognition"]["micro_rate"] == summary["micro_recognition_rate"]
    report = _write_json(tmp_path / "closeout.json", payload)
    assert audit_project_closeout(report, root=tmp_path) == payload

    paths["readme_path"].write_text("drift", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact drifted: readme"):
        audit_project_closeout(report, root=tmp_path)
