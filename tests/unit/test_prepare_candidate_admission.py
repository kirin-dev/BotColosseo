import json
from dataclasses import asdict
from pathlib import Path

import torch

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.cli.prepare_candidate_admission import (
    derive_candidate_metrics,
    prepare_candidate_admission,
)
from botcolosseo.training.historical_pool import (
    HistoricalPoolManifest,
    PoolEntry,
    write_pool_atomic,
)
from botcolosseo.training.league_checkpoint import (
    LeagueCheckpointState,
    LeagueRunIdentity,
)


def _grid(ids: tuple[str, ...], default: float) -> dict[str, dict[str, float]]:
    return {left: {right: default for right in ids} for left in ids}


def test_candidate_metrics_use_current_pool_axes_and_five_script_opponents() -> None:
    active = ("m2-anchor", "policy-0200000")
    scripts = tuple(f"script-{index}" for index in range(5))
    policies = (*active, "candidate-0400000", *scripts)
    wins = _grid(policies, 0.5)
    objectives = _grid(policies, 0.5)
    wins["candidate-0400000"].update(
        {"m2-anchor": 0.7, "policy-0200000": 0.4}
    )
    wins["policy-0200000"].update(
        {"m2-anchor": 0.6, "policy-0200000": 0.5}
    )
    for index, script in enumerate(scripts):
        wins["candidate-0400000"][script] = 0.6 + index * 0.05
        objectives["candidate-0400000"][script] = 0.8

    metrics = derive_candidate_metrics(
        {"win_rate": wins, "objective_rate": objectives},
        candidate_id="candidate-0400000",
        active_ids=active,
        script_ids=scripts,
    )

    assert metrics["candidate_payoffs"] == {
        "m2-anchor": 0.7,
        "policy-0200000": 0.4,
    }
    assert metrics["active_payoffs"]["policy-0200000"] == {
        "m2-anchor": 0.6,
        "policy-0200000": 0.5,
    }
    assert metrics["candidate_historical_worst_case"] == 0.4
    assert metrics["active_historical_worst_case"] == 0.5
    assert metrics["candidate_script_average"] == 0.7
    assert metrics["candidate_script_worst_case"] == 0.6
    assert metrics["candidate_objective_rate"] == 0.8


def test_prepare_candidate_writes_hash_bound_entry_metrics_and_selection_report(
    tmp_path: Path,
) -> None:
    base = tmp_path / "runs/m2/base.pt"
    base.parent.mkdir(parents=True)
    base.write_text("base", encoding="utf-8")
    validation = tmp_path / "reports/m2/validation/summary.json"
    validation.parent.mkdir(parents=True)
    validation.write_text("{}", encoding="utf-8")
    anchor = PoolEntry(
        policy_id="m2-anchor",
        checkpoint="runs/m2/base.pt",
        checkpoint_sha256=sha256_file(base),
        scenario_hash="6" * 64,
        config_hash="2" * 64,
        source_git_commit="a" * 40,
        parent_checkpoint_sha256=sha256_file(base),
        environment_steps=800_000,
        admitted_at_utc="2026-07-21T00:00:00Z",
        validation_report="reports/m2/validation/summary.json",
        validation_report_sha256=sha256_file(validation),
        script_average_win_rate=0.77,
        script_worst_case_win_rate=0.23,
        objective_rate=0.93,
        payoff_by_policy={"m2-anchor": 0.5},
        anchor=True,
        admission_reason="anchor",
    )
    pool = HistoricalPoolManifest(
        1, 0, None, "2026-07-21T00:00:00Z", (anchor,)
    )
    pool_path = tmp_path / "reports/m3/pool-v0.json"
    write_pool_atomic(pool, pool_path)
    identity = LeagueRunIdentity(
        base_checkpoint_sha256=sha256_file(base),
        config_hash="2" * 64,
        train_manifest_hash="3" * 64,
        pool_manifest_hash=pool.manifest_sha256,
        payoff_report_hash="5" * 64,
        scenario_hash="6" * 64,
    )
    candidate = tmp_path / "runs/m3/candidate.pt"
    candidate.parent.mkdir(parents=True)
    torch.save(
        {
            "schema_version": 1,
            "identity": asdict(identity),
            "state": asdict(LeagueCheckpointState(200_000, 10, 2, 1)),
        },
        candidate,
    )
    scripts = (
        "random_legal",
        "fixed_route",
        "objective_first",
        "aggressive_script",
        "defensive_script",
    )
    policies = ("m2-anchor", "candidate-0200000", *scripts)
    wins = _grid(policies, 0.5)
    objectives = _grid(policies, 0.8)
    matrix = {
        "policy_ids": list(policies),
        "win_rate": wins,
        "objective_rate": objectives,
    }
    matrix_dir = tmp_path / "reports/m3/validation/candidate-0200000/matrix"
    matrix_dir.mkdir(parents=True)
    matrix_path = matrix_dir / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    csv_path = matrix_dir / "crossplay.csv"
    csv_path.write_text("raw\n", encoding="utf-8")
    manifest = {
        "split": "validation",
        "test_cases_accessed": False,
        "pool_manifest_sha256": pool.manifest_sha256,
        "candidate_checkpoint_sha256": sha256_file(candidate),
        "matrix_sha256": sha256_file(matrix_path),
        "crossplay_csv_sha256": sha256_file(csv_path),
        "protocol_inconsistencies": 0,
        "executed_rows": 5 * len(policies) * (len(policies) + 1),
    }
    (matrix_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    report = tmp_path / "reports/m3/validation/candidate-0200000/report.json"
    entry_path = report.with_name("entry.json")
    metrics_path = report.with_name("metrics.json")

    entry, metrics = prepare_candidate_admission(
        artifact_root=tmp_path,
        pool_path=pool_path,
        candidate_checkpoint=candidate,
        candidate_id="candidate-0200000",
        matrix_dir=matrix_dir,
        output_entry=entry_path,
        output_metrics=metrics_path,
        output_candidate_report=report,
        source_git_commit="b" * 40,
        admitted_at_utc="2026-07-21T01:00:00Z",
    )

    assert entry.validation_report_sha256 == sha256_file(report)
    assert entry.payoff_by_policy == {"m2-anchor": 0.5}
    assert metrics.source_split == "validation"
    assert json.loads(entry_path.read_text(encoding="utf-8"))["anchor"] is False
    assert json.loads(report.read_text(encoding="utf-8"))["test_cases_accessed"] is False
