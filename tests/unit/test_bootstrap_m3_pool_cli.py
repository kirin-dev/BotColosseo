import json
from pathlib import Path

import torch

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.agents.model import AsymmetricActorCritic
from botcolosseo.cli.bootstrap_m3_pool import bootstrap_initial_pool
from botcolosseo.training.historical_pool import load_pool


def test_bootstrap_creates_hash_bound_m2_anchor_and_payoff(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs/m2/selected.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "schema_version": 1,
            "model": AsymmetricActorCritic().state_dict(),
            "metadata": {
                "config_hash": "config",
                "scenario_hash": "scenario",
                "counters": {"environment_steps": 800_000},
            },
        },
        checkpoint,
    )
    evidence = tmp_path / "reports/m3/bootstrap/anchor"
    evidence.mkdir(parents=True)
    summary = {
        "official": False,
        "protocol_inconsistencies": 0,
        "artifact_inconsistencies": 0,
        "policies": {
            "ppo": {
                "wins": {"rate": 0.8},
                "objectives": {"rate": 0.9},
                "opponents": {
                    "a": {"wins": {"rate": 0.7}},
                    "b": {"wins": {"rate": 0.6}},
                },
            }
        },
    }
    summary_path = evidence / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest = {
        "split": "validation",
        "official": False,
        "checkpoint_sha256": {"ppo": sha256_file(checkpoint)},
        "summary_sha256": sha256_file(summary_path),
        "git_commit": "a" * 40,
        "scenario_hash": "scenario",
    }
    (evidence / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pool_path = tmp_path / "reports/m3/pool.json"
    payoff_path = tmp_path / "reports/m3/payoffs.json"

    bootstrap_initial_pool(
        artifact_root=tmp_path,
        checkpoint=checkpoint,
        validation_evidence_dir=evidence,
        output_pool=pool_path,
        output_payoffs=payoff_path,
        policy_id="m2-anchor",
        admitted_at_utc="2026-07-21T00:00:00Z",
        audited_checkpoint_sha256=sha256_file(checkpoint),
    )

    pool = load_pool(pool_path, artifact_root=tmp_path)
    payoffs = json.loads(payoff_path.read_text(encoding="utf-8"))
    anchor = pool.entries[0]
    assert anchor.anchor is True
    assert anchor.environment_steps == 800_000
    assert anchor.script_average_win_rate == 0.8
    assert anchor.script_worst_case_win_rate == 0.6
    assert anchor.objective_rate == 0.9
    assert payoffs["pool_manifest_sha256"] == pool.manifest_sha256
    assert payoffs["win_rates"] == {"m2-anchor": 0.5}


def test_bootstrap_refuses_checkpoint_not_authorized_by_m2_audit(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    try:
        bootstrap_initial_pool(
            artifact_root=tmp_path,
            checkpoint=checkpoint,
            validation_evidence_dir=tmp_path / "missing",
            output_pool=tmp_path / "pool.json",
            output_payoffs=tmp_path / "payoffs.json",
            policy_id="m2-anchor",
            admitted_at_utc="2026-07-21T00:00:00Z",
            audited_checkpoint_sha256="0" * 64,
        )
    except ValueError as error:
        assert "audit" in str(error)
    else:
        raise AssertionError("unaudited checkpoint was accepted")
