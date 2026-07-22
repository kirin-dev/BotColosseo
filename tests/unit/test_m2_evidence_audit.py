from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from botcolosseo.cli.audit_m2_evidence import main as audit_main
from botcolosseo.evaluation import m2_evidence_audit
from botcolosseo.evaluation.m2_evidence_audit import (
    audit_official_evidence,
    audit_repository_provenance,
)
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS

POLICIES = ("ppo", "bc", "random_legal")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(report_dir: Path) -> None:
    report_dir.mkdir()
    rows = []
    for policy in POLICIES:
        for opponent_index, opponent in enumerate(DUEL_OPPONENTS):
            for side in ("host", "opponent"):
                rows.append(
                    {
                        "policy": policy,
                        "opponent": opponent,
                        "pair_index": opponent_index,
                        "learner_side": side,
                        "seed": 10_000 + opponent_index,
                    }
                )
    episodes = report_dir / "episodes.csv"
    with episodes.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gates = {
        "official": True,
        "complete": True,
        "protocol_clean": True,
        "artifact_clean": True,
        "ppo_win_rate_minus_bc": True,
        "ppo_win_rate_minus_random": True,
        "ppo_objective_rate_minus_bc": True,
        "paired_score_lcb_positive": True,
        "per_opponent_floor": True,
    }
    summary = report_dir / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "official": True,
                "complete": True,
                "passed": True,
                "episodes": len(rows),
                "expected_episodes": len(rows),
                "protocol_inconsistencies": 0,
                "artifact_inconsistencies": 0,
                "environment_retries": 1,
                "gates": gates,
                "policies": {policy: {} for policy in POLICIES},
            }
        ),
        encoding="utf-8",
    )
    for policy in ("ppo", "bc"):
        (report_dir / f"{policy}-training-summary.json").write_text(
            json.dumps({"selected_checkpoint_sha256": f"{policy}-hash"}),
            encoding="utf-8",
        )
    manifest = {
        "official": True,
        "split": "test",
        "git_dirty": False,
        "checkpoint_sha256": {"ppo": "ppo-hash", "bc": "bc-hash"},
        "policies": list(POLICIES),
        "opponents": list(DUEL_OPPONENTS),
        "pairs_per_opponent": 1,
        "episodes_sha256": _sha256(episodes),
        "summary_sha256": _sha256(summary),
    }
    (report_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _refresh_episode_hash(report_dir: Path) -> None:
    manifest_path = report_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["episodes_sha256"] = _sha256(report_dir / "episodes.csv")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_official_evidence_audit_accepts_exact_paired_artifacts(tmp_path: Path) -> None:
    report_dir = tmp_path / "m2"
    _write_fixture(report_dir)

    result = audit_official_evidence(report_dir, pairs_per_opponent=1)

    assert result["episodes"] == 30
    assert result["pair_groups"] == 15
    assert result["environment_retries"] == 1
    assert result["passed"] is True


def test_official_evidence_audit_rejects_side_imbalance(tmp_path: Path) -> None:
    report_dir = tmp_path / "m2"
    _write_fixture(report_dir)
    episodes = report_dir / "episodes.csv"
    text = episodes.read_text()
    episodes.write_text(text.replace(",opponent,10000", ",opponent,99999", 1))
    _refresh_episode_hash(report_dir)

    with pytest.raises(ValueError, match="side-swapped"):
        audit_official_evidence(report_dir, pairs_per_opponent=1)


def test_official_evidence_audit_rejects_failed_gate(tmp_path: Path) -> None:
    report_dir = tmp_path / "m2"
    _write_fixture(report_dir)
    summary_path = report_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["passed"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path = report_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["summary_sha256"] = _sha256(summary_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="did not pass"):
        audit_official_evidence(report_dir, pairs_per_opponent=1)


def test_integrity_only_audit_preserves_failed_capability_result(tmp_path: Path) -> None:
    report_dir = tmp_path / "m2"
    _write_fixture(report_dir)
    summary_path = report_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["passed"] = False
    summary["gates"]["ppo_win_rate_minus_bc"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path = report_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["summary_sha256"] = _sha256(summary_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = audit_official_evidence(
        report_dir, pairs_per_opponent=1, require_capability_pass=False
    )

    assert result["integrity_passed"] is True
    assert result["capability_passed"] is False
    assert result["passed"] is False


def test_audit_cli_can_report_intentionally_pending_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert audit_main(["--report-dir", str(tmp_path), "--allow-pending"]) == 0
    assert '"official_status": "pending"' in capsys.readouterr().out


def test_audit_cli_cannot_weaken_the_frozen_pair_count(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        audit_main(
            [
                "--report-dir",
                str(tmp_path),
                "--allow-pending",
                "--pairs-per-opponent",
                "1",
            ]
        )


def test_repository_provenance_rechecks_tracked_inputs_and_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    report_dir = root / "reports/m2"
    report_dir.parent.mkdir(parents=True)
    _write_fixture(report_dir)
    (root / "configs/m2").mkdir(parents=True)
    (root / "assets/scenarios/crystal_run").mkdir(parents=True)
    checkpoint_paths = {
        "ppo": root / "runs/m2/ppo-full/selected.pt",
        "bc": root / "runs/m2/bc-full/best.pt",
    }
    for policy, checkpoint in checkpoint_paths.items():
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"{policy}-checkpoint".encode())
    config = root / "configs/m2/evaluation.yaml"
    config.write_text(
        "policies:\n"
        "  ppo:\n    checkpoint: runs/m2/ppo-full/selected.pt\n"
        "  bc:\n    checkpoint: runs/m2/bc-full/best.pt\n"
    )
    split = root / "configs/m2/test.json"
    split.write_text("[]\n")
    scenario = root / "assets/scenarios/crystal_run/manifest.json"
    scenario.write_text(json.dumps({"wad_sha256": "scenario-hash"}))
    manifest_path = report_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "checkpoint_sha256": {
                policy: _sha256(path) for policy, path in checkpoint_paths.items()
            },
            "config_sha256": _sha256(config),
            "split_sha256": _sha256(split),
            "scenario_manifest_sha256": _sha256(scenario),
            "scenario_hash": "scenario-hash",
            "git_commit": "a" * 40,
        }
    )
    manifest_path.write_text(json.dumps(manifest))
    for policy in ("ppo", "bc"):
        (report_dir / f"{policy}-training-summary.json").write_text(
            json.dumps(
                {"selected_checkpoint_sha256": manifest["checkpoint_sha256"][policy]}
            )
        )
    monkeypatch.setattr(
        m2_evidence_audit.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    result = audit_official_evidence(report_dir, pairs_per_opponent=1)
    result = audit_repository_provenance(root, report_dir, result)

    assert result["repository_provenance"] is True
    assert result["evaluation_commit"] == "a" * 40
