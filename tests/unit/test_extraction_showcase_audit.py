from __future__ import annotations

import json
from pathlib import Path

import pytest

from botcolosseo.cli.audit_extraction_showcase import (
    _manifest_tier,
    audit_extraction_showcase,
)
from botcolosseo.data.demonstrations import sha256_file

POLICIES = ("strong", "aggressive", "defensive", "explorer")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _claims(policy: str) -> dict[str, object]:
    claims = {
        "decisions": 300,
        "died": False,
        "extracted": True,
        "extracted_value": 85,
        "valid_hits": 0,
        "kills": 0,
        "cache_looted": 0,
        "aggressive_chains": 0,
        "successful_disengagements": 0,
        "meaningful_extractions": 1,
        "meaningful_loot_regions": 1,
        "backpack_upgrades": 0,
        "upgrade_to_extraction_conversions": 0,
    }
    if policy == "aggressive":
        claims.update(
            valid_hits=5,
            kills=1,
            cache_looted=1,
            aggressive_chains=1,
        )
    elif policy == "defensive":
        claims["successful_disengagements"] = 1
    elif policy == "explorer":
        claims.update(
            meaningful_loot_regions=4,
            backpack_upgrades=1,
            upgrade_to_extraction_conversions=1,
        )
    return claims


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    selections = {}
    board_artifacts = {}
    for index, policy in enumerate(POLICIES):
        checkpoint_sha256 = f"{index}" * 64
        manifest_path = tmp_path / f"{policy}-manifest.json"
        manifest = {
            "policy": policy,
            "gate_schema_version": 2,
            "eligible": True,
            "selected_checkpoint_sha256": checkpoint_sha256,
            "test_cases_accessed": False,
        }
        _write_json(manifest_path, manifest)
        video = tmp_path / f"{policy}.mp4"
        video.write_bytes(f"{policy}-video".encode())
        evidence_path = tmp_path / f"{policy}-evidence.json"
        evidence = {
            "policy": policy,
            "case_index": index,
            "checkpoint_sha256": checkpoint_sha256,
            "media": video.name,
            "media_sha256": sha256_file(video),
            "showcase_claims": _claims(policy),
            "render_attempt_count": 1,
            "render_attempts": [{"attempt": 1, "accepted": True}],
            "frame_count": 300,
            "fps": 10,
            "test_cases_accessed": False,
        }
        _write_json(evidence_path, evidence)
        selections[policy] = {
            "case_index": index,
            "checkpoint_sha256": checkpoint_sha256,
            "paired_style_difference": 0 if policy == "strong" else 1,
            "artifact_manifest": manifest_path.name,
            "artifact_manifest_sha256": sha256_file(manifest_path),
            "evidence_tier": "research_selection",
        }
        board_artifacts[policy] = {
            "video": video.name,
            "video_sha256": sha256_file(video),
            "evidence": evidence_path.name,
            "evidence_sha256": sha256_file(evidence_path),
            "evidence_tier": "research_selection",
        }
    selection_path = tmp_path / "selection.json"
    _write_json(
        selection_path,
        {
            "schema_version": 2,
            "selection_split": "validation",
            "selections": selections,
            "test_cases_accessed": False,
        },
    )
    board_path = tmp_path / "board.png"
    board_path.write_bytes(b"board")
    board_manifest_path = tmp_path / "board-manifest.json"
    _write_json(
        board_manifest_path,
        {
            "schema_version": 1,
            "board": board_path.name,
            "board_sha256": sha256_file(board_path),
            "selection": selection_path.name,
            "selection_sha256": sha256_file(selection_path),
            "artifacts": board_artifacts,
            "source_split": "validation",
            "test_cases_accessed": False,
        },
    )
    method_path = tmp_path / "method.svg"
    method_path.write_text("<svg/>", encoding="utf-8")
    return selection_path, board_manifest_path, method_path


def test_product_showcase_audit_accepts_bound_representative_media(
    tmp_path: Path,
) -> None:
    selection, board, method = _artifacts(tmp_path)

    result = audit_extraction_showcase(
        root=tmp_path,
        selection_path=selection,
        board_manifest_path=board,
        method_path=method,
    )

    assert result["passed"] is True
    assert result["official_test_eligible"] is False
    assert result["policies"]["explorer"]["story_checks_passed"] is True
    assert result["policies"]["explorer"]["research_gate_passed"] is True


def test_product_showcase_audit_rejects_claim_drift(tmp_path: Path) -> None:
    selection, board, method = _artifacts(tmp_path)
    board_payload = json.loads(board.read_text(encoding="utf-8"))
    explorer_evidence = tmp_path / board_payload["artifacts"]["explorer"]["evidence"]
    evidence = json.loads(explorer_evidence.read_text(encoding="utf-8"))
    evidence["showcase_claims"]["upgrade_to_extraction_conversions"] = 0
    _write_json(explorer_evidence, evidence)
    board_payload["artifacts"]["explorer"]["evidence_sha256"] = sha256_file(
        explorer_evidence
    )
    _write_json(board, board_payload)

    with pytest.raises(ValueError, match="story is incomplete"):
        audit_extraction_showcase(
            root=tmp_path,
            selection_path=selection,
            board_manifest_path=board,
            method_path=method,
        )


def test_showcase_audit_accepts_product_only_strong_manifest() -> None:
    checkpoint_sha256 = "a" * 64
    manifest = {
        "policy": "strong",
        "admission_kind": "strong_product_showcase",
        "product_showcase_eligible": True,
        "research_gate_passed": False,
        "official_test_eligible": False,
        "showcase_checkpoint_sha256": checkpoint_sha256,
        "test_cases_accessed": False,
    }

    assert (
        _manifest_tier("strong", checkpoint_sha256, manifest)
        == "product_showcase"
    )


def test_showcase_audit_accepts_representative_case_manifest() -> None:
    checkpoint_sha256 = "a" * 64
    manifest = {
        "policy": "explorer",
        "evidence_tier": "representative_case_demonstration",
        "product_demo_eligible": True,
        "research_gate_passed": False,
        "official_test_eligible": False,
        "claim_scope": "representative_validation_cases_only",
        "aggregate_style_gate_passed": False,
        "representative_case_count": 3,
        "showcase_checkpoint_sha256": checkpoint_sha256,
        "test_cases_accessed": False,
    }

    assert (
        _manifest_tier("explorer", checkpoint_sha256, manifest)
        == "representative_case_demonstration"
    )
