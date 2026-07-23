from dataclasses import replace

import pytest

from botcolosseo.evaluation.defensive import (
    DEFENSIVE_POLICIES,
    PROTECTIVE_PRESENCE_ESTIMATOR,
    DefensiveEpisodeRecord,
    evaluate_defensive_records,
    paired_cluster_bootstrap_ratio_difference,
)
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS


def _record(
    policy: str, opponent: str, side: str, *, pair_index: int
) -> DefensiveEpisodeRecord:
    defensive = policy == "defensive"
    return DefensiveEpisodeRecord(
        policy=policy,
        split="validation",
        opponent=opponent,
        pair_index=pair_index,
        seed=pair_index + 7,
        learner_side=side,
        outcome="win",
        objective_completed=True,
        learner_score=3,
        opponent_score=0,
        decisions=100,
        risk_decisions=20,
        protective_presence_decisions=16 if defensive else 4,
        carrier_opportunities=10,
        carrier_denials=2 if defensive else 0,
        recovery_opportunities=10,
        recoveries=2 if defensive else 0,
        no_risk_decisions=80,
        unnecessary_guard_decisions=8 if defensive else 4,
        low_health_opportunities=2,
        successful_escapes=1 if defensive else 0,
        terminated=True,
        truncated=False,
        peer_tic_lag_max=0,
        protocol_inconsistent=False,
        action_tic_inconsistent=False,
        score_event_inconsistent=False,
        scenario_hash="scenario",
    )


def _records() -> list[DefensiveEpisodeRecord]:
    return [
        _record(policy, opponent, side, pair_index=index)
        for index, opponent in enumerate(DUEL_OPPONENTS)
        for policy in DEFENSIVE_POLICIES
        for side in ("host", "opponent")
    ]


def test_defensive_summary_passes_style_retention_and_integrity_gates() -> None:
    summary = evaluate_defensive_records(
        _records(),
        expected_pairs_per_opponent=1,
        expected_scenario_hash="scenario",
        bootstrap_seed=3,
        bootstrap_samples=500,
    )

    assert summary.complete is True
    assert summary.passed is True
    assert summary.skill_retention == 1.0
    assert summary.protective_presence_estimator == PROTECTIVE_PRESENCE_ESTIMATOR
    assert summary.protective_presence_delta == pytest.approx(0.6)
    assert summary.protective_presence_delta_ci[0] > 0
    assert summary.policies["defensive"].denial_recovery_rate == 0.2


def test_camping_and_missing_event_opportunities_fail_defensive_gates() -> None:
    records = [
        replace(
            row,
            carrier_opportunities=0,
            carrier_denials=0,
            recovery_opportunities=0,
            recoveries=0,
            unnecessary_guard_decisions=40,
        )
        if row.policy == "defensive"
        else row
        for row in _records()
    ]

    summary = evaluate_defensive_records(
        records,
        expected_pairs_per_opponent=1,
        expected_scenario_hash="scenario",
        bootstrap_seed=3,
        bootstrap_samples=100,
    )

    assert summary.gates["denial_recovery_improved"] is False
    assert summary.gates["unnecessary_guard_controlled"] is False
    assert summary.passed is False


def test_duplicate_or_protocol_error_fails_defensive_integrity() -> None:
    records = _records()
    records[-1] = replace(records[-2], protocol_inconsistent=True)

    summary = evaluate_defensive_records(
        records,
        expected_pairs_per_opponent=1,
        expected_scenario_hash="scenario",
        bootstrap_seed=3,
        bootstrap_samples=100,
    )

    assert summary.complete is False
    assert summary.protocol_inconsistencies > 0
    assert summary.gates["protocol_clean"] is False


def test_paired_ratio_bootstrap_does_not_treat_zero_risk_as_failure() -> None:
    counts = [
        (0, 0, 38, 40),
        (50, 100, 40, 100),
    ]

    first = paired_cluster_bootstrap_ratio_difference(counts, seed=11, samples=500)
    second = paired_cluster_bootstrap_ratio_difference(counts, seed=11, samples=500)

    assert first == second
    assert first is not None
    point, interval = first
    assert point == pytest.approx(0.5 - 78 / 140)
    assert point > -0.1
    assert interval[1] >= point


def test_paired_ratio_bootstrap_fails_closed_without_policy_opportunities() -> None:
    assert (
        paired_cluster_bootstrap_ratio_difference(
            [(0, 0, 4, 10)], seed=3, samples=100
        )
        is None
    )
