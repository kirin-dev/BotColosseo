from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from botcolosseo.evaluation.showcase import (
    build_showcase_manifest,
    canonical_json,
    case_id,
    load_metric_evidence,
    load_showcase_cases,
    load_showcase_config,
    publish_staged_files,
    select_highlight_window,
    select_showcase_case,
    write_jsonl,
)


def _metric_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage": "m4",
        "split": "validation",
        "passed": True,
        "style_gate_passed": True,
        "retention_gate_passed": True,
        "episodes": 800,
        "checkpoint_sha256": {"strong_base": "1" * 64, "aggressive": "2" * 64},
        "headline": {
            "base_win_rate": 0.72,
            "aggressive_style_delta": 0.31,
            "skill_retention": 0.89,
        },
        "case_contrast_scores": {"fixed_route:250:host": 0.5},
        "decision_contrast_scores": {
            "fixed_route:250:host": [0.0, 1.0, 0.0]
        },
    }


def _eligible_record(case_id: str, policy_id: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "policy_id": policy_id,
        "terminated": True,
        "truncated": False,
        "objective_completed": True,
        "environment_attempts": 1,
        "peer_tic_lag_max": 0,
        "protocol_inconsistent": False,
        "action_tic_inconsistent": False,
        "score_event_inconsistent": False,
    }


def test_development_config_is_non_public_and_hash_bound() -> None:
    config = load_showcase_config(
        Path("configs/showcase/development.yaml"), root=Path.cwd()
    )

    assert config.stage == "development"
    assert config.publication is False
    assert [policy.policy_id for policy in config.policies] == ["ppo", "bc"]
    assert config.metrics_path is None
    assert config.render.gif_max_bytes == 10_000_000
    assert config.output_dir == Path.cwd() / "artifacts/showcase-development/media"


def test_m4_cases_are_validation_only_and_cover_four_scripts() -> None:
    cases = load_showcase_cases(
        Path("configs/showcase/m4-validation.json"),
        root=Path.cwd(),
        expected_count=8,
    )

    assert {case.split for case in cases} == {"validation"}
    assert {case.learner_side for case in cases} == {"host", "opponent"}
    assert {case.opponent for case in cases} == {
        "fixed_route",
        "objective_first",
        "aggressive_script",
        "defensive_script",
    }
    assert len({case_id(case) for case in cases}) == 8


def test_case_manifest_rejects_source_hash_drift(tmp_path: Path) -> None:
    (tmp_path / "validation.json").write_text("[]\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "source": "validation.json",
        "source_sha256": "0" * 64,
        "cases": [],
    }
    path = tmp_path / "showcase.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="source hash"):
        load_showcase_cases(path, root=tmp_path, expected_count=0)


def test_publication_config_rejects_test_split(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "stage": "m4",
        "publication": True,
        "split": "test",
        "cases": "configs/showcase/m4-validation.json",
        "metrics": "reports/m4/validation/summary.json",
        "policies": [
            {
                "id": "strong_base",
                "label": "Strong Base",
                "checkpoint": "runs/m3/selected.pt",
                "expected_sha256": "1" * 64,
            },
            {
                "id": "aggressive",
                "label": "Aggressive",
                "checkpoint": "runs/m4/aggressive/selected.pt",
                "expected_sha256": "2" * 64,
            },
        ],
        "render": {
            "fps": 10,
            "gif_seconds": 18,
            "gif_max_bytes": 10_000_000,
            "max_decisions": 525,
            "output_dir": "docs/assets/showcase",
        },
        "evidence_dir": "reports/showcase/m4",
    }
    path = tmp_path / "m4.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="validation"):
        load_showcase_config(path, root=Path.cwd())


def test_metric_evidence_is_stage_and_checkpoint_hash_bound(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(_metric_payload()), encoding="utf-8")

    evidence = load_metric_evidence(
        path,
        expected_stage="m4",
        expected_hashes={"strong_base": "1" * 64, "aggressive": "2" * 64},
    )

    assert evidence.skill_retention == 0.89
    assert evidence.case_contrast_scores["fixed_route:250:host"] == 0.5


def test_metric_evidence_accepts_m5_hashes_only_for_m5_stage(tmp_path: Path) -> None:
    payload = _metric_payload()
    payload["stage"] = "m5"
    payload["checkpoint_sha256"] = {
        "strong_base": "1" * 64,
        "aggressive": "2" * 64,
        "defensive": "3" * 64,
        "explorer": "4" * 64,
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    hashes = payload["checkpoint_sha256"]

    with pytest.raises(ValueError, match="stage"):
        load_metric_evidence(path, expected_stage="m4", expected_hashes=hashes)

    evidence = load_metric_evidence(path, expected_stage="m5", expected_hashes=hashes)

    assert evidence.checkpoint_sha256 == hashes


def test_select_showcase_case_uses_contrast_then_case_id_and_rejects_truncation() -> None:
    policy_ids = ("strong_base", "aggressive")
    records = [
        _eligible_record("zulu", policy_id) for policy_id in policy_ids
    ] + [
        _eligible_record("alpha", policy_id) for policy_id in policy_ids
    ] + [
        _eligible_record("truncated", policy_id) for policy_id in policy_ids
    ]
    records[-1]["truncated"] = True

    selection = select_showcase_case(
        records,
        policy_ids=policy_ids,
        contrast_scores={"zulu": 0.5, "alpha": 0.5, "truncated": 1.0},
    )

    assert selection.selected_case_id == "alpha"
    assert [record["policy_id"] for record in selection.selected_records] == list(
        policy_ids
    )
    assert selection.ranking == (("alpha", 0.5), ("zulu", 0.5))
    assert "truncated" in selection.rejection_reasons


def test_select_showcase_case_rejects_when_every_case_is_ineligible() -> None:
    records = [_eligible_record("only", "strong_base"), _eligible_record("only", "aggressive")]
    records[0]["truncated"] = True

    with pytest.raises(ValueError, match="No showcase case satisfies publication eligibility"):
        select_showcase_case(
            records,
            policy_ids=("strong_base", "aggressive"),
            contrast_scores={"only": 0.5},
        )


def test_development_selection_keeps_protocol_gates_but_allows_time_limit() -> None:
    records = [
        _eligible_record("only", "ppo"),
        _eligible_record("only", "bc"),
    ]
    records[1]["terminated"] = False
    records[1]["truncated"] = True

    selection = select_showcase_case(
        records,
        policy_ids=("ppo", "bc"),
        contrast_scores={"only": 0.0},
        require_normal_termination=False,
    )

    assert selection.selected_case_id == "only"

    records[1]["protocol_inconsistent"] = True
    with pytest.raises(ValueError, match="No showcase case"):
        select_showcase_case(
            records,
            policy_ids=("ppo", "bc"),
            contrast_scores={"only": 0.0},
            require_normal_termination=False,
        )


def test_select_highlight_window_breaks_equal_totals_at_earliest_window() -> None:
    assert select_highlight_window(
        (0.0, 1.0, 1.0, 0.0, 1.0), window_frames=3
    ) == (0, 3)


def test_canonical_json_and_jsonl_are_stable(tmp_path: Path) -> None:
    assert canonical_json({"z": 1, "a": 2}) == b'{"a":2,"z":1}\n'

    output = write_jsonl(tmp_path / "episodes.jsonl", ({"z": 1, "a": 2},))

    assert output.read_bytes() == b'{"a":2,"z":1}\n'


def test_manifest_is_hash_bound_and_labels_validation_only(tmp_path: Path) -> None:
    root = Path.cwd()
    config = load_showcase_config(
        Path("configs/showcase/development.yaml"), root=root
    )
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_bytes(b'{"episode":1}\n')
    media = (
        {
            "path": "artifacts/showcase-development/media/development-comparison.gif",
            "sha256": "4" * 64,
            "bytes": 1234,
            "frame_count": 180,
            "dimensions": [332, 512],
            "fps": 10,
        },
    )

    manifest = build_showcase_manifest(
        git_commit="a" * 40,
        git_dirty=False,
        config=config,
        scenario_hash="1" * 64,
        case_manifest_sha256="2" * 64,
        checkpoint_sha256={"ppo": "3" * 64, "bc": "4" * 64},
        metric_sha256=None,
        episodes_path=episodes,
        selected_case="random_legal:250:host",
        highlight=(0, 180),
        media=media,
        gate_passed=False,
    )

    assert manifest["split"] == "validation"
    assert manifest["official_test_result"] is False
    assert manifest["test_cases_accessed"] is False
    assert manifest["episode_log_sha256"] == hashlib.sha256(
        episodes.read_bytes()
    ).hexdigest()
    assert len(manifest["run_identity"]) == 64
    assert manifest["media"] == list(media)


def test_publication_replaces_manifest_last_and_rejects_identity_drift(
    tmp_path: Path,
) -> None:
    staged_media = tmp_path / "staged.gif"
    staged_media.write_bytes(b"GIF89a")
    target_media = tmp_path / "published/showcase.gif"
    staged_manifest = tmp_path / "staged-manifest.json"
    staged_manifest.write_bytes(canonical_json({"run_identity": "a" * 64}))
    target_manifest = tmp_path / "published/manifest.json"

    publish_staged_files(
        ((staged_media, target_media),),
        staged_manifest=staged_manifest,
        target_manifest=target_manifest,
        run_identity="a" * 64,
    )

    assert target_media.read_bytes() == b"GIF89a"
    assert json.loads(target_manifest.read_text())["run_identity"] == "a" * 64

    staged_manifest.write_bytes(canonical_json({"run_identity": "b" * 64}))
    with pytest.raises(ValueError, match="identity"):
        publish_staged_files(
            ((staged_media, target_media),),
            staged_manifest=staged_manifest,
            target_manifest=target_manifest,
            run_identity="b" * 64,
        )
