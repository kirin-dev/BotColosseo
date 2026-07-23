from dataclasses import replace

import pytest

from botcolosseo.evaluation.explorer import (
    EXPLORER_POLICIES,
    ROUTE_ENTROPY_ESTIMATOR,
    ExplorerEpisodeRecord,
    classify_completed_route,
    evaluate_explorer_records,
    normalized_route_entropy,
)
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS


def _record(
    policy: str,
    opponent: str,
    side: str,
    *,
    pair_index: int,
) -> ExplorerEpisodeRecord:
    explorer = policy == "explorer"
    return ExplorerEpisodeRecord(
        policy=policy,
        split="validation",
        opponent=opponent,
        pair_index=pair_index,
        seed=pair_index + 11,
        learner_side=side,
        outcome="win",
        objective_completed=True,
        learner_score=3,
        opponent_score=0,
        learner_scores=3,
        decisions=300 if explorer else 250,
        upper_completions=1 if explorer else 3,
        lower_completions=1 if explorer else 0,
        flank_completions=1 if explorer else 0,
        mixed_or_unknown_completions=0,
        unique_regions=8 if explorer else 4,
        terminated=True,
        truncated=False,
        peer_tic_lag_max=0,
        protocol_inconsistent=False,
        action_tic_inconsistent=False,
        score_event_inconsistent=False,
        scenario_hash="scenario",
    )


def _records() -> list[ExplorerEpisodeRecord]:
    return [
        _record(policy, opponent, side, pair_index=index)
        for index, opponent in enumerate(DUEL_OPPONENTS)
        for policy in EXPLORER_POLICIES
        for side in ("host", "opponent")
    ]


def test_route_attribution_and_entropy_are_task_bound() -> None:
    assert classify_completed_route(("center", "upper_route", "home")) == "upper"
    assert classify_completed_route(("center", "lower_route", "home")) == "lower"
    assert (
        classify_completed_route(("center", "flank_east", "flank_west", "home"))
        == "flank"
    )
    assert classify_completed_route(("center", "home")) == "mixed_or_unknown"
    assert normalized_route_entropy((3, 0, 0)) == 0.0
    assert normalized_route_entropy((1, 1, 1)) == pytest.approx(1.0)


def test_explorer_summary_passes_style_retention_and_efficiency_gates() -> None:
    summary = evaluate_explorer_records(
        _records(),
        expected_pairs_per_opponent=1,
        expected_scenario_hash="scenario",
        bootstrap_seed=7,
        bootstrap_samples=500,
    )

    assert summary.complete is True
    assert summary.passed is True
    assert summary.skill_retention == 1.0
    assert summary.route_entropy_estimator == ROUTE_ENTROPY_ESTIMATOR
    assert summary.route_entropy_delta == pytest.approx(1.0)
    assert summary.route_entropy_delta_ci[0] > 0
    assert summary.policies["explorer"].flank_completion_rate == pytest.approx(1 / 3)


def test_explorer_wandering_and_missing_flank_fail_hard_gates() -> None:
    records = [
        replace(
            row,
            decisions=500,
            upper_completions=2,
            lower_completions=1,
            flank_completions=0,
        )
        if row.policy == "explorer"
        else row
        for row in _records()
    ]

    summary = evaluate_explorer_records(
        records,
        expected_pairs_per_opponent=1,
        expected_scenario_hash="scenario",
        bootstrap_seed=7,
        bootstrap_samples=100,
    )

    assert summary.gates["route_coverage"] is False
    assert summary.gates["flank_improved"] is False
    assert summary.gates["efficiency_controlled"] is False
    assert summary.passed is False


def test_explorer_duplicate_or_protocol_error_fails_integrity() -> None:
    records = _records()
    records[-1] = replace(records[-2], protocol_inconsistent=True)

    summary = evaluate_explorer_records(
        records,
        expected_pairs_per_opponent=1,
        expected_scenario_hash="scenario",
        bootstrap_seed=7,
        bootstrap_samples=100,
    )

    assert summary.complete is False
    assert summary.protocol_inconsistencies > 0
    assert summary.gates["protocol_clean"] is False
