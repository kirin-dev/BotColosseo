from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from botcolosseo.evaluation.m2 import (
    M2EpisodeRecord,
    evaluate_m2_records,
    paired_bootstrap_interval,
    valid_action_tic_boundary,
    validate_paired_schedule,
    wilson_interval,
)
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS, DuelCase


def _cases(*, pairs: int = 2, split: str = "test") -> tuple[DuelCase, ...]:
    rows: list[DuelCase] = []
    pair_index = 0
    for opponent in DUEL_OPPONENTS:
        for local_pair in range(pairs):
            seed = 10_000 + pair_index
            for side in ("host", "opponent"):
                rows.append(
                    DuelCase(
                        split=split,
                        pair_index=pair_index,
                        seed=seed,
                        opponent=opponent,
                        learner_side=side,
                        core_spawn_index=local_pair % 3,
                        route="direct_upper",
                    )
                )
            pair_index += 1
    return tuple(rows)


def _record(
    case: DuelCase,
    policy: str,
    *,
    outcome: str,
    objective: bool,
    score_difference: int,
) -> M2EpisodeRecord:
    learner_score = max(score_difference, 0)
    opponent_score = max(-score_difference, 0)
    return M2EpisodeRecord(
        policy=policy,
        split=case.split,
        opponent=case.opponent,
        pair_index=case.pair_index,
        seed=case.seed,
        learner_side=case.learner_side,
        outcome=outcome,
        objective_completed=objective,
        learner_score=learner_score,
        opponent_score=opponent_score,
        decisions=20,
        terminated=True,
        truncated=False,
        peer_tic_lag_max=0,
        protocol_inconsistent=False,
        scenario_hash="scenario",
    )


def _passing_records() -> list[M2EpisodeRecord]:
    records: list[M2EpisodeRecord] = []
    for case in _cases(pairs=50):
        index = case.pair_index % 10
        records.extend(
            (
                _record(
                    case,
                    "ppo",
                    outcome="win" if index < 7 else "loss",
                    objective=index < 8,
                    score_difference=2 if index < 7 else -1,
                ),
                _record(
                    case,
                    "bc",
                    outcome="win" if index < 5 else "loss",
                    objective=index < 6,
                    score_difference=1 if index < 5 else -2,
                ),
                _record(
                    case,
                    "random_legal",
                    outcome="win" if index < 3 else "loss",
                    objective=index < 4,
                    score_difference=1 if index < 3 else -3,
                ),
            )
        )
    return records


def _summary(records: list[M2EpisodeRecord], *, official: bool = True):
    return evaluate_m2_records(
        records,
        official=official,
        expected_pairs_per_opponent=50,
        bootstrap_seed=17,
        bootstrap_samples=2_000,
        expected_scenario_hash="scenario",
    )


def test_schedule_requires_adjacent_side_swapped_pairs() -> None:
    cases = _cases()
    validate_paired_schedule(cases, expected_split="test", pairs_per_opponent=2)

    bad = list(cases)
    bad[1] = replace(bad[1], learner_side="host")
    with pytest.raises(ValueError, match="side-swapped"):
        validate_paired_schedule(bad, expected_split="test", pairs_per_opponent=2)


def test_intervals_are_deterministic_and_known() -> None:
    assert wilson_interval(75, 100) == pytest.approx((0.65696, 0.82455), abs=1e-5)
    differences = np.array([0.5, 1.0, 1.5, 2.0])
    first = paired_bootstrap_interval(differences, seed=7, samples=2_000)
    second = paired_bootstrap_interval(differences, seed=7, samples=2_000)
    assert first == second
    assert first[0] > 0.0


def test_only_terminal_boundary_may_advance_fewer_than_four_action_tics() -> None:
    assert valid_action_tic_boundary(4, terminated=False, truncated=False)
    assert valid_action_tic_boundary(0, terminated=True, truncated=False)
    assert valid_action_tic_boundary(3, terminated=False, truncated=True)
    assert not valid_action_tic_boundary(3, terminated=False, truncated=False)
    assert not valid_action_tic_boundary(5, terminated=True, truncated=False)


def test_complete_official_result_passes_every_frozen_gate() -> None:
    summary = _summary(_passing_records())

    assert summary.complete is True
    assert summary.passed is True
    assert summary.gates["ppo_win_rate_minus_bc"] is True
    assert summary.gates["ppo_win_rate_minus_random"] is True
    assert summary.gates["ppo_objective_rate_minus_bc"] is True
    assert summary.gates["paired_score_lcb_positive"] is True
    assert summary.gates["per_opponent_floor"] is True
    assert summary.policies["ppo"].episodes == 500


def test_each_performance_gate_can_independently_block_pass() -> None:
    records = _passing_records()
    ppo_objective_indices = [
        index
        for index, record in enumerate(records)
        if record.policy == "ppo" and record.objective_completed
    ]
    objective_failure = list(records)
    for index in ppo_objective_indices[:150]:
        objective_failure[index] = replace(
            objective_failure[index], objective_completed=False
        )
    assert _summary(objective_failure).gates["ppo_objective_rate_minus_bc"] is False

    score_failure = [
        replace(
            record,
            learner_score=(record.opponent_score + 1),
            outcome="win",
        )
        if record.policy == "bc"
        else record
        for record in records
    ]
    failed = _summary(score_failure)
    assert failed.gates["ppo_win_rate_minus_bc"] is False
    assert failed.gates["paired_score_lcb_positive"] is False

    random_failure = [
        replace(
            record,
            learner_score=(record.opponent_score + 1),
            outcome="win",
        )
        if record.policy == "random_legal"
        else record
        for record in records
    ]
    assert _summary(random_failure).gates["ppo_win_rate_minus_random"] is False

    floor_failure: list[M2EpisodeRecord] = []
    for record in records:
        if record.opponent == "defensive_script" and record.policy == "ppo":
            local_index = record.pair_index % 10
            if local_index in (4, 5, 6):
                record = replace(
                    record,
                    learner_score=record.opponent_score - 1,
                    outcome="loss",
                )
        floor_failure.append(record)
    floor_summary = _summary(floor_failure)
    assert floor_summary.gates["per_opponent_floor"] is False


def test_partial_or_dirty_result_cannot_pass() -> None:
    cases = _cases(pairs=1, split="validation")
    records = [
        _record(case, policy, outcome="win", objective=True, score_difference=3)
        for case in cases
        for policy in ("ppo", "bc", "random_legal")
    ]
    records[0] = replace(records[0], protocol_inconsistent=True)

    summary = evaluate_m2_records(
        records,
        official=False,
        expected_pairs_per_opponent=50,
        bootstrap_seed=17,
        bootstrap_samples=100,
        expected_scenario_hash="scenario",
    )

    assert summary.official is False
    assert summary.complete is False
    assert summary.protocol_inconsistencies == 1
    assert summary.passed is False


def test_full_counts_without_matching_side_pairs_are_incomplete() -> None:
    records = _passing_records()
    first_ppo = next(
        index for index, record in enumerate(records) if record.policy == "ppo"
    )
    records[first_ppo] = replace(records[first_ppo], pair_index=99_999)

    summary = _summary(records)

    assert summary.episodes == 1500
    assert summary.complete is False
    assert summary.passed is False


def test_artifact_inconsistency_blocks_an_otherwise_passing_gate() -> None:
    summary = evaluate_m2_records(
        _passing_records(),
        official=True,
        expected_pairs_per_opponent=50,
        bootstrap_seed=17,
        bootstrap_samples=500,
        expected_scenario_hash="scenario",
        artifact_inconsistencies=1,
    )

    assert summary.gates["artifact_clean"] is False
    assert summary.passed is False
