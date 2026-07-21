import json
from dataclasses import asdict
from pathlib import Path

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.cli.update_historical_pool import main
from botcolosseo.training.historical_pool import (
    AdmissionMetrics,
    HistoricalPoolManifest,
    PoolEntry,
    load_pool,
    write_pool_atomic,
)


def _artifact(root: Path, relative: str, content: str) -> tuple[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return relative, sha256_file(path)


def _entry(root: Path, policy_id: str, *, anchor: bool, payoff: float) -> PoolEntry:
    checkpoint, checkpoint_hash = _artifact(
        root, f"runs/m3/{policy_id}.pt", f"checkpoint-{policy_id}"
    )
    report, report_hash = _artifact(
        root, f"reports/m3/validation/{policy_id}.json", f"report-{policy_id}"
    )
    return PoolEntry(
        policy_id=policy_id,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        scenario_hash="scenario",
        config_hash="config",
        source_git_commit="a" * 40,
        parent_checkpoint_sha256="b" * 64,
        environment_steps=0 if anchor else 200_000,
        admitted_at_utc="2026-07-21T00:00:00Z",
        validation_report=report,
        validation_report_sha256=report_hash,
        script_average_win_rate=0.75,
        script_worst_case_win_rate=0.60,
        objective_rate=0.90,
        payoff_by_policy={"policy-anchor": payoff},
        anchor=anchor,
        admission_reason="anchor" if anchor else "candidate",
    )


def _inputs(tmp_path: Path, *, integrity_ok: bool) -> tuple[Path, Path, Path]:
    anchor = _entry(tmp_path, "policy-anchor", anchor=True, payoff=0.0)
    pool = HistoricalPoolManifest(1, 0, None, "2026-07-21T00:00:00Z", (anchor,))
    pool_path = tmp_path / "reports/m3/pool.json"
    write_pool_atomic(pool, pool_path)
    candidate = _entry(tmp_path, "policy-candidate", anchor=False, payoff=1.0)
    entry_path = tmp_path / "reports/m3/candidate-entry.json"
    entry_path.write_text(
        json.dumps(asdict(candidate), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = AdmissionMetrics(
        integrity_ok=integrity_ok,
        validation_complete=True,
        paired_side_swapped=True,
        protocol_inconsistencies=0,
        source_split="validation",
        candidate_script_average=0.75,
        active_script_average=0.75,
        candidate_historical_worst_case=0.65,
        active_historical_worst_case=0.50,
        candidate_payoffs={"policy-anchor": 1.0},
        active_payoffs={"policy-anchor": {"policy-anchor": 0.0}},
    )
    metrics_path = tmp_path / "reports/m3/admission-metrics.json"
    metrics_path.write_text(
        json.dumps(asdict(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return pool_path, entry_path, metrics_path


def test_pool_update_emits_decision_report_even_when_rejected(tmp_path: Path) -> None:
    pool, entry, metrics = _inputs(tmp_path, integrity_ok=False)
    output_pool = tmp_path / "reports/m3/pool-v1.json"
    decision = tmp_path / "reports/m3/admission-decision.json"

    result = main(
        [
            "--artifact-root",
            str(tmp_path),
            "--pool",
            str(pool),
            "--entry",
            str(entry),
            "--metrics",
            str(metrics),
            "--output-pool",
            str(output_pool),
            "--decision-report",
            str(decision),
        ]
    )

    report = json.loads(decision.read_text(encoding="utf-8"))
    assert result == 2
    assert report["eligible"] is False
    assert report["reason"] == "integrity audit failed"
    assert report["hashes_verified"] is True
    assert not output_pool.exists()


def test_pool_update_verifies_all_artifacts_and_writes_new_version(tmp_path: Path) -> None:
    pool, entry, metrics = _inputs(tmp_path, integrity_ok=True)
    output_pool = tmp_path / "reports/m3/pool-v1.json"
    decision = tmp_path / "reports/m3/admission-decision.json"
    payoffs = tmp_path / "reports/m3/payoffs-v1.json"

    assert (
        main(
            [
                "--artifact-root",
                str(tmp_path),
                "--pool",
                str(pool),
                "--entry",
                str(entry),
                "--metrics",
                str(metrics),
                "--output-pool",
                str(output_pool),
                "--decision-report",
                str(decision),
                "--output-payoffs",
                str(payoffs),
            ]
        )
        == 0
    )
    updated = load_pool(output_pool, artifact_root=tmp_path)
    report = json.loads(decision.read_text(encoding="utf-8"))
    assert updated.pool_version == 1
    assert len(updated.entries) == 2
    assert report["eligible"] is True
    assert report["new_pool_manifest_sha256"] == updated.manifest_sha256
    payoff_report = json.loads(payoffs.read_text(encoding="utf-8"))
    assert payoff_report == {
        "pool_manifest_sha256": updated.manifest_sha256,
        "schema_version": 1,
        "split": "validation",
        "win_rates": {"policy-anchor": 1.0, "policy-candidate": 0.5},
    }
    assert report["new_payoff_report_sha256"] == sha256_file(payoffs)
