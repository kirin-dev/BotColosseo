from __future__ import annotations

import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from botcolosseo.cli.admit_extraction_strong_showcase import (
    build_strong_showcase_admission,
)
from botcolosseo.cli.create_extraction_strong_demonstration import (
    build_strong_demonstration,
)
from botcolosseo.cli.resolve_extraction_strong_artifact import (
    resolve_strong_artifact,
)
from botcolosseo.data.demonstrations import sha256_file
from tests.unit.test_extraction_gates import episodes


def _write_report(
    path: Path,
    *,
    split: str,
    checkpoint: Path,
    episode_items: tuple[object, ...],
) -> None:
    payload = {
        "policy": "strong",
        "policy_kind": "strong-recurrent-ppo",
        "metric_schema_version": 2,
        "split": split,
        "complete": True,
        "test_cases_accessed": False,
        "actor_privilege_violations": 0,
        "fair_actor_observation_only": True,
        "checkpoint": str(checkpoint.relative_to(path.parent.parent.parent.parent)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "protocol_sha256": "p" * 64,
        "scenario_hash": "s" * 64,
        "metrics": {"episodes": [asdict(item) for item in episode_items]},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _evidence(tmp_path: Path, *, heldout_extracted: int = 82) -> tuple[Path, Path]:
    output = tmp_path / "runs/extraction/strong-ppo"
    evaluation = output / "evaluation-v2"
    evaluation.mkdir(parents=True)
    checkpoint = output / "candidate-0400000.pt"
    checkpoint.write_bytes(b"strong-400k")
    validation_path = evaluation / "candidate-0400000-validation.json"
    heldout_path = evaluation / "candidate-0400000-heldout.json"
    solo_path = evaluation / "candidate-0400000-solo.json"
    validation = episodes(240)
    heldout = tuple(
        replace(item, extracted=index < heldout_extracted)
        for index, item in enumerate(episodes(120))
    )
    solo = tuple(
        replace(item, opponent_style="idle") for item in episodes(40)
    )
    _write_report(
        validation_path,
        split="validation",
        checkpoint=checkpoint,
        episode_items=validation,
    )
    _write_report(
        heldout_path,
        split="heldout",
        checkpoint=checkpoint,
        episode_items=heldout,
    )
    _write_report(
        solo_path,
        split="solo",
        checkpoint=checkpoint,
        episode_items=solo,
    )
    relative_checkpoint = str(checkpoint.relative_to(tmp_path))
    score = [5, 0.88, 0.83, 0.94, 71.5, 0]
    candidate = {
        "checkpoint": relative_checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint),
        "eligible": True,
        "report": str(validation_path.relative_to(tmp_path)),
        "report_sha256": sha256_file(validation_path),
        "score": score,
    }
    ranking = {
        "schema_version": 1,
        "policy": "strong",
        "selection_split": "validation",
        "test_cases_accessed": False,
        "candidates": [candidate],
        "pareto_frontier": [
            {
                key: candidate[key]
                for key in ("checkpoint", "checkpoint_sha256", "eligible", "score")
            }
        ],
    }
    ranking_path = evaluation / "ranking.json"
    ranking_path.write_text(json.dumps(ranking), encoding="utf-8")
    return ranking_path, evaluation


def test_strong_product_admission_preserves_research_failure_and_resolves(
    tmp_path: Path,
) -> None:
    ranking, evaluation = _evidence(tmp_path)

    result, checkpoint = build_strong_showcase_admission(
        root=tmp_path,
        ranking_path=ranking,
        evaluation_root=evaluation,
    )

    assert checkpoint.name == "candidate-0400000.pt"
    assert result["research_failed_checks"] == ["heldout_extraction"]
    assert result["product_showcase_eligible"] is True
    assert result["official_test_eligible"] is False
    output = tmp_path / "runs/extraction/strong-ppo"
    showcase = output / "showcase.pt"
    manifest = output / "showcase-admission.json"
    shutil.copyfile(checkpoint, showcase)
    result["showcase_checkpoint"] = str(showcase.relative_to(tmp_path))
    result["showcase_checkpoint_sha256"] = sha256_file(showcase)
    manifest.write_text(json.dumps(result), encoding="utf-8")

    resolved = resolve_strong_artifact(
        tmp_path, Path("runs/extraction/strong-ppo")
    )

    assert resolved["mode"] == "product_showcase"
    assert resolved["checkpoint"] == "runs/extraction/strong-ppo/showcase.pt"
    assert resolved["heldout_report"].endswith("-heldout.json")


def test_strong_product_admission_rejects_excessive_heldout_gap(
    tmp_path: Path,
) -> None:
    ranking, evaluation = _evidence(tmp_path, heldout_extracted=76)

    with pytest.raises(ValueError, match="product Showcase rule"):
        build_strong_showcase_admission(
            root=tmp_path,
            ranking_path=ranking,
            evaluation_root=evaluation,
        )


def test_direct_strong_demonstration_discloses_research_failures(
    tmp_path: Path,
) -> None:
    _, evaluation = _evidence(tmp_path)
    checkpoint = evaluation.parent / "candidate-0400000.pt"

    result = build_strong_demonstration(
        root=tmp_path,
        checkpoint=checkpoint,
        validation_path=evaluation / "candidate-0400000-validation.json",
        heldout_path=evaluation / "candidate-0400000-heldout.json",
        solo_path=evaluation / "candidate-0400000-solo.json",
    )

    assert result["product_showcase_eligible"] is True
    assert result["research_gate_passed"] is False
    assert result["research_failed_checks"] == ["heldout_extraction"]
    assert result["claim_scope"] == "product_showcase_capability_only"
