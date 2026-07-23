import json
from pathlib import Path

import pytest
import torch

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.agents.model import AsymmetricActorCritic
from botcolosseo.agents.style_model import RoutedStyledActorCritic
from botcolosseo.evaluation.m5_v2_training_audit import audit_m5_v2_training


def _evidence(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / "latest.pt"
    model = RoutedStyledActorCritic.from_base(
        AsymmetricActorCritic(), bottleneck=8
    )
    torch.save(
        {
            "schema_version": 1,
            "identity": {
                "base_checkpoint_sha256": "a" * 64,
                "config_hash": "b" * 64,
                "train_manifest_hash": "c" * 64,
                "pool_manifest_hash": "d" * 64,
                "payoff_report_hash": "e" * 64,
                "scenario_hash": "scenario",
            },
            "state": {
                "environment_steps": 2_000,
                "updates": 1,
                "episodes": 0,
                "next_pair_slot": 0,
            },
            "model": model.state_dict(),
        },
        checkpoint,
    )
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "environment_steps": 2_000,
        "scenario_hash": "scenario",
        "style": "explorer",
        "style_reward_components": {
            "mode:0:target_region": 1.0,
            "mode:1:target_region": 1.0,
            "mode:2:target_region": 1.0,
        },
        "supervision_counts": {
            "tokens": 30,
            "mode:0": 10,
            "mode:1": 10,
            "mode:2": 10,
        },
        "teacher": "route_explorer_teacher",
        "test_cases_accessed": False,
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    train = {
        "kind": "train",
        "total_loss": 1.0,
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 1.0,
        "approximate_kl": 0.01,
        "pre_clip_grad_norm": 0.5,
        "post_clip_grad_norm": 0.4,
        "style_base_kl": 0.02,
        "auxiliary_loss": 1.5,
        "teacher_agreement": 0.5,
    }
    (run / "metrics.jsonl").write_text(
        json.dumps(train) + "\n", encoding="utf-8"
    )
    return run


def test_m5_v2_audit_accepts_complete_routed_evidence(tmp_path: Path) -> None:
    result = audit_m5_v2_training(
        _evidence(tmp_path), style="explorer", expected_steps=2_000
    )

    assert result["passed"] is True
    assert result["train_updates_audited"] == 1


def test_m5_v2_audit_rejects_missing_route_mode(tmp_path: Path) -> None:
    run = _evidence(tmp_path)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    del summary["supervision_counts"]["mode:2"]
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="every route mode"):
        audit_m5_v2_training(run, style="explorer", expected_steps=2_000)
