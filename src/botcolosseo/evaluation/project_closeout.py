from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from botcolosseo.agents.league_opponents import sha256_file


def build_project_closeout(
    *,
    root: Path,
    synthetic_summary_path: Path,
    synthetic_provenance_path: Path,
    synthetic_responses_path: Path,
    curation_manifest_path: Path,
    study_manifest_path: Path,
    release_record_path: Path,
    policy_archive_path: Path,
    study_archive_path: Path,
    curated_archive_path: Path,
    future_plan_path: Path,
    readme_path: Path,
    readme_cn_path: Path,
) -> dict[str, object]:
    root = root.resolve()
    paths = {
        "synthetic_summary": synthetic_summary_path,
        "synthetic_provenance": synthetic_provenance_path,
        "synthetic_responses": synthetic_responses_path,
        "curation_manifest": curation_manifest_path,
        "study_manifest": study_manifest_path,
        "release_record": release_record_path,
        "policy_archive": policy_archive_path,
        "study_archive": study_archive_path,
        "curated_archive": curated_archive_path,
        "future_plan": future_plan_path,
        "readme": readme_path,
        "readme_cn": readme_cn_path,
    }
    resolved = {name: path.resolve() for name, path in paths.items()}
    if any(
        not path.is_relative_to(root) or not path.is_file()
        for path in resolved.values()
    ):
        raise ValueError("Project closeout source path is invalid")
    summary = _json_object(resolved["synthetic_summary"])
    provenance = _json_object(resolved["synthetic_provenance"])
    curation = _json_object(resolved["curation_manifest"])
    study = _json_object(resolved["study_manifest"])
    release = _json_object(resolved["release_record"])
    _validate_sources(
        summary=summary,
        provenance=provenance,
        curation=curation,
        study=study,
        release=release,
        responses_sha256=sha256_file(resolved["synthetic_responses"]),
        study_manifest_sha256=sha256_file(resolved["study_manifest"]),
        policy_archive_sha256=sha256_file(resolved["policy_archive"]),
    )
    per_style = summary["per_style"]
    assert isinstance(per_style, Mapping)
    return {
        "schema_version": 1,
        "stage": "current-scenario-project-closeout",
        "current_scenario": "Crystal Run Arena",
        "current_scenario_development_complete": True,
        "product_release_audit_passed": True,
        "synthetic_preflight_passed": True,
        "synthetic_data": True,
        "human_participants": False,
        "human_user_study_completed": False,
        "human_user_study_claimed": False,
        "future_extraction_scenario_status": "proposal_only",
        "recognition": {
            "simulated_respondents": summary["respondents"],
            "responses": summary["responses"],
            "macro_rate": summary["macro_recognition_rate"],
            "micro_rate": summary["micro_recognition_rate"],
            "per_style": {
                style: per_style[style]["recognition_rate"]
                for style in ("aggressive", "defensive", "explorer")
            },
        },
        "artifacts": {
            name: _artifact(root, path)
            for name, path in resolved.items()
        },
        "evidence_boundary": (
            "Engineering-complete current-scenario release with a synthetic "
            "perception preflight; no human-study claim and no extraction-v2 result."
        ),
        "test_cases_accessed": False,
    }


def audit_project_closeout(
    report_path: Path,
    *,
    root: Path,
) -> dict[str, object]:
    root = root.resolve()
    payload = _json_object(report_path.resolve())
    if (
        payload.get("schema_version") != 1
        or payload.get("stage") != "current-scenario-project-closeout"
        or payload.get("current_scenario_development_complete") is not True
        or payload.get("product_release_audit_passed") is not True
        or payload.get("synthetic_preflight_passed") is not True
        or payload.get("synthetic_data") is not True
        or payload.get("human_participants") is not False
        or payload.get("human_user_study_completed") is not False
        or payload.get("human_user_study_claimed") is not False
        or payload.get("future_extraction_scenario_status") != "proposal_only"
        or payload.get("test_cases_accessed") is not False
    ):
        raise ValueError("Project closeout identity is invalid")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("Project closeout artifact map is invalid")
    for name, row in artifacts.items():
        if not isinstance(name, str) or not isinstance(row, Mapping):
            raise ValueError("Project closeout artifact row is invalid")
        relative = row.get("path")
        if not isinstance(relative, str):
            raise ValueError("Project closeout artifact path is invalid")
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or row.get("sha256") != sha256_file(path)
            or row.get("bytes") != path.stat().st_size
        ):
            raise ValueError(f"Project closeout artifact drifted: {name}")
    return payload


def _validate_sources(
    *,
    summary: Mapping[str, object],
    provenance: Mapping[str, object],
    curation: Mapping[str, object],
    study: Mapping[str, object],
    release: Mapping[str, object],
    responses_sha256: str,
    study_manifest_sha256: str,
    policy_archive_sha256: str,
) -> None:
    per_style = summary.get("per_style")
    if (
        summary.get("stage") != "m6-user-study-analysis"
        or summary.get("synthetic_data") is not True
        or summary.get("human_participants") is not False
        or summary.get("respondents") != 10
        or summary.get("responses") != 60
        or _rate(summary, "macro_recognition_rate") < 0.8
        or _rate(summary, "micro_recognition_rate") < 0.8
        or not isinstance(per_style, Mapping)
        or any(
            not isinstance(per_style.get(style), Mapping)
            or _rate(per_style[style], "recognition_rate") < 0.75
            for style in ("aggressive", "defensive", "explorer")
        )
        or summary.get("test_cases_accessed") is not False
    ):
        raise ValueError("Project closeout synthetic analysis is invalid")
    if (
        provenance.get("stage") != "m6-synthetic-user-study-preflight"
        or provenance.get("synthetic_data") is not True
        or provenance.get("human_participants") is not False
        or provenance.get("respondent_count") != 10
        or provenance.get("response_count") != 60
        or provenance.get("responses_sha256") != responses_sha256
        or summary.get("responses_sha256") != responses_sha256
        or provenance.get("test_cases_accessed") is not False
    ):
        raise ValueError("Project closeout synthetic provenance is invalid")
    if (
        curation.get("stage") != "m6-curated-validation-clips"
        or curation.get("clip_count") != 6
        or curation.get("clips_per_style") != 2
        or curation.get("test_cases_accessed") is not False
        or study.get("stage") != "m6-user-study-package"
        or study.get("clip_count") != 6
        or study.get("clips_per_style") != 2
        or study.get("assignment_count") != 10
        or study.get("test_cases_accessed") is not False
        or summary.get("package_manifest_sha256") != study_manifest_sha256
    ):
        raise ValueError("Project closeout study package is invalid")
    if (
        release.get("stage") != "hybrid-product-release"
        or release.get("audit_passed") is not True
        or release.get("test_cases_accessed") is not False
        or release.get("archive_sha256") != policy_archive_sha256
    ):
        raise ValueError("Project closeout policy release is invalid")


def _artifact(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _rate(payload: Mapping[str, object], field: str) -> float:
    value = payload.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= float(value) <= 1
    ):
        raise ValueError(f"Project closeout {field} is invalid")
    return float(value)


def _json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload
