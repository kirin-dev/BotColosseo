import json
from dataclasses import replace
from pathlib import Path

import pytest

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.cli.select_strong_base import main
from botcolosseo.evaluation.strong_base_selection import (
    StrongBaseCandidate,
    load_candidate_evidence,
    select_strong_base,
)


def _candidate(
    policy_id: str,
    *,
    integrity_passed: bool = True,
    historical: float = 0.60,
    script: float = 0.70,
    objective: float = 0.80,
    steps: int = 200_000,
) -> StrongBaseCandidate:
    return StrongBaseCandidate(
        policy_id=policy_id,
        checkpoint=f"runs/m3/{policy_id}.pt",
        checkpoint_sha256=(policy_id[-1] * 64),
        validation_report=f"reports/m3/validation/{policy_id}.json",
        validation_report_sha256=(policy_id[-1] * 64),
        environment_steps=steps,
        integrity_passed=integrity_passed,
        rejection_reasons=() if integrity_passed else ("protocol inconsistency",),
        historical_worst_case_win_rate=historical,
        script_average_win_rate=script,
        full_objective_rate=objective,
        config_hash="a" * 64,
        pool_manifest_sha256="b" * 64,
        payoff_report_sha256="c" * 64,
        scenario_hash="scenario",
    )


def test_selection_uses_exact_lexicographic_order_and_reports_rejections() -> None:
    candidates = (
        _candidate("policy-1", integrity_passed=False, historical=1.0),
        _candidate("policy-2", historical=0.61, script=0.60, objective=0.70),
        _candidate("policy-3", historical=0.61, script=0.71, objective=0.70),
        _candidate("policy-4", historical=0.61, script=0.71, objective=0.81, steps=400_000),
        _candidate("policy-5", historical=0.61, script=0.71, objective=0.81, steps=200_000),
    )

    decision = select_strong_base(candidates)

    assert decision.selected.policy_id == "policy-5"
    assert decision.selection_rule == (
        "integrity_passed_desc",
        "historical_worst_case_win_rate_desc",
        "script_average_win_rate_desc",
        "full_objective_rate_desc",
        "environment_steps_asc",
    )
    rejected = {row["policy_id"]: row for row in decision.candidates}
    assert rejected["policy-1"]["eligible"] is False
    assert rejected["policy-1"]["rejection_reasons"] == ["protocol inconsistency"]


def test_selection_rejects_nonfinite_metrics_and_exact_unresolved_ties() -> None:
    with pytest.raises(ValueError, match="finite probabilities"):
        replace(_candidate("policy-1"), script_average_win_rate=float("nan"))
    with pytest.raises(ValueError, match="exact tie"):
        select_strong_base((_candidate("policy-1"), _candidate("policy-2")))


@pytest.mark.parametrize(
    "relative",
    (
        "reports/m3/test/candidate.json",
        "reports/m3/heldout/candidate.json",
        "reports/m3/held-out-candidate.json",
    ),
)
def test_candidate_evidence_rejects_test_or_heldout_paths(
    tmp_path: Path, relative: str
) -> None:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="validation-only"):
        load_candidate_evidence(path, artifact_root=tmp_path)


def test_selection_cli_verifies_hashes_and_writes_canonical_report(
    tmp_path: Path,
) -> None:
    reports = []
    for index, candidate in enumerate(
        (_candidate("policy-1"), _candidate("policy-2", historical=0.65)), start=1
    ):
        checkpoint = tmp_path / candidate.checkpoint
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(f"checkpoint-{index}", encoding="utf-8")
        report = tmp_path / candidate.validation_report
        report.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "policy_id": candidate.policy_id,
            "checkpoint": candidate.checkpoint,
            "checkpoint_sha256": sha256_file(checkpoint),
            "environment_steps": candidate.environment_steps,
            "split": "validation",
            "test_cases_accessed": False,
            "integrity_passed": candidate.integrity_passed,
            "rejection_reasons": list(candidate.rejection_reasons),
            "historical_worst_case_win_rate": candidate.historical_worst_case_win_rate,
            "script_average_win_rate": candidate.script_average_win_rate,
            "full_objective_rate": candidate.full_objective_rate,
            "config_hash": candidate.config_hash,
            "pool_manifest_sha256": candidate.pool_manifest_sha256,
            "payoff_report_sha256": candidate.payoff_report_sha256,
            "scenario_hash": candidate.scenario_hash,
        }
        report.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        reports.append(report)
    output = tmp_path / "reports/m3/selection.json"

    assert (
        main(
            [
                "--artifact-root",
                str(tmp_path),
                "--candidate-report",
                str(reports[0]),
                "--candidate-report",
                str(reports[1]),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["selected"]["policy_id"] == "policy-2"
    assert result["split"] == "validation"
    assert result["test_cases_accessed"] is False
    assert result["candidate_reports"] == sorted(
        str(path.relative_to(tmp_path)) for path in reports
    )
