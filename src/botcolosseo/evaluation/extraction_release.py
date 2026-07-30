from __future__ import annotations

import json
from pathlib import Path

from botcolosseo.data.demonstrations import sha256_file

STYLES = ("strong", "aggressive", "defensive", "explorer")
RELEASE_ARTIFACTS = (
    "assets/scenarios/crystal_run_extraction/crystal_run_extraction.cfg",
    "assets/scenarios/crystal_run_extraction/crystal_run_extraction.wad",
    "assets/scenarios/crystal_run_extraction/manifest.json",
    "configs/extraction_v2/showcase-candidates.json",
    "configs/extraction_v2/aggressive-showcase-search.json",
    "docs/assets/extraction-v2/showcase-board.png",
    "reports/extraction-v2/x0-mechanics.json",
    "reports/extraction-v2/training-artifact-audit.json",
    "reports/extraction-v2/showcase-candidates-strong.json",
    "reports/extraction-v2/showcase-candidates-defensive.json",
    "reports/extraction-v2/showcase-candidates-explorer.json",
    "reports/extraction-v2/aggressive-showcase-search.json",
)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_for_case(
    report: dict[str, object],
    *,
    seed: int,
    learner_side: str,
) -> dict[str, object]:
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    episodes = metrics["episodes"]
    assert isinstance(episodes, list)
    matches = [
        episode
        for episode in episodes
        if episode["seed"] == seed and episode["learner_side"] == learner_side
    ]
    if len(matches) != 1:
        raise ValueError("Showcase case is not unique in its selection report")
    return matches[0]


def _artifact_record(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing Extraction v2 release artifact: {relative}")
    return {
        "bytes": path.stat().st_size,
        "path": relative,
        "sha256": sha256_file(path),
    }


def build_extraction_release(root: Path) -> dict[str, object]:
    root = root.resolve()
    style_records: dict[str, dict[str, object]] = {}
    evidence_by_style: dict[str, dict[str, object]] = {}
    artifacts = list(RELEASE_ARTIFACTS)
    for style in STYLES:
        evidence_path = f"reports/extraction-v2/showcase/{style}.json"
        media_path = f"docs/assets/extraction-v2/{style}.mp4"
        evidence = _read_json(root / evidence_path)
        evidence_by_style[style] = evidence
        if evidence["style"] != style:
            raise ValueError(f"{style} showcase evidence has the wrong style")
        if evidence["test_cases_accessed"] or evidence["case"]["split"] != "validation":
            raise ValueError(f"{style} showcase is not validation-only")
        if evidence["media"] != media_path:
            raise ValueError(f"{style} showcase points at the wrong media")
        if sha256_file(root / media_path) != evidence["media_sha256"]:
            raise ValueError(f"{style} showcase media hash drifted")
        duration = evidence["frame_count"] / evidence["fps"]
        if not 30 <= duration <= 45:
            raise ValueError(f"{style} showcase duration is outside 30-45 seconds")
        style_records[style] = {
            "case": evidence["case"],
            "checkpoint_sha256": evidence["checkpoint_sha256"],
            "duration_seconds": round(duration, 3),
            "evidence": evidence_path,
            "extracted_value": evidence["extracted_value"],
            "media": media_path,
            "media_sha256": evidence["media_sha256"],
            "policy_kind": evidence["policy_kind"],
        }
        if evidence.get("base_checkpoint_sha256") is not None:
            style_records[style]["base_checkpoint_sha256"] = evidence[
                "base_checkpoint_sha256"
            ]
        artifacts.extend((evidence_path, media_path))

    strong = evidence_by_style["strong"]
    aggressive = evidence_by_style["aggressive"]
    defensive = evidence_by_style["defensive"]
    explorer = evidence_by_style["explorer"]
    defensive_case = _episode_for_case(
        _read_json(root / "reports/extraction-v2/showcase-candidates-defensive.json"),
        seed=defensive["case"]["seed"],
        learner_side=defensive["case"]["learner_side"],
    )
    explorer_case = _episode_for_case(
        _read_json(root / "reports/extraction-v2/showcase-candidates-explorer.json"),
        seed=explorer["case"]["seed"],
        learner_side=explorer["case"]["learner_side"],
    )
    learner = explorer["case"]["learner_side"]
    acceptance = {
        "aggressive_five_hit_kill_cache_extract": bool(
            aggressive["showcase_claims"]["aggressive_chain_complete"]
        ),
        "all_clips_validation_only": all(
            not item["test_cases_accessed"] for item in evidence_by_style.values()
        ),
        "all_clips_within_30_to_45_seconds": True,
        "defensive_zero_attack_meaningful_extraction": bool(
            defensive["extracted_value"] >= 25
            and defensive_case["attack_decisions"] == 0
            and defensive_case["extracted"]
        ),
        "explorer_route_and_backpack_upgrade": bool(
            explorer_case["unique_route_cells"] >= 25
            and explorer["events"].get(f"{learner}:loot_drop", 0) >= 1
            and explorer["extracted_value"] >= 50
        ),
        "strong_generalist_extraction": bool(
            strong["extracted"] and strong["extracted_value"] >= 50
        ),
        "no_test_case_access": True,
    }
    failed = [name for name, passed in acceptance.items() if passed is not True]
    if failed:
        raise ValueError(f"Extraction v2 showcase acceptance failed: {failed}")
    scenario = _read_json(
        root / "assets/scenarios/crystal_run_extraction/manifest.json"
    )
    return {
        "acceptance": acceptance,
        "artifacts": [
            _artifact_record(root, relative) for relative in sorted(set(artifacts))
        ],
        "scenario_hash": scenario["wad_sha256"],
        "schema_version": 1,
        "split": "validation",
        "stage": "crystal-run-extraction-v2-showcase",
        "styles": style_records,
        "test_cases_accessed": False,
    }


def audit_extraction_release(report_path: Path, *, root: Path) -> dict[str, object]:
    root = root.resolve()
    payload = _read_json(report_path)
    if payload["stage"] != "crystal-run-extraction-v2-showcase":
        raise ValueError("Unexpected Extraction v2 release stage")
    if payload["test_cases_accessed"]:
        raise ValueError("Extraction v2 release accessed test cases")
    if not all(payload["acceptance"].values()):
        raise ValueError("Extraction v2 release contains a failed acceptance claim")
    for artifact in payload["artifacts"]:
        path = root / artifact["path"]
        if path.stat().st_size != artifact["bytes"]:
            raise ValueError(f"Extraction v2 artifact size drift: {artifact['path']}")
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Extraction v2 artifact hash drift: {artifact['path']}")
    return payload
