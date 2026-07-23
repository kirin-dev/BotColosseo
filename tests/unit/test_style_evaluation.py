from __future__ import annotations

from dataclasses import replace

from botcolosseo.evaluation.style import (
    STYLE_POLICIES,
    StyleEpisodeRecord,
    evaluate_aggressive_records,
)
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS


def _record(
    policy: str,
    opponent: str,
    side: str,
    *,
    pair_index: int,
) -> StyleEpisodeRecord:
    aggressive = policy == "aggressive"
    return StyleEpisodeRecord(
        policy=policy,
        split="validation",
        opponent=opponent,
        pair_index=pair_index,
        seed=pair_index + 10,
        learner_side=side,
        outcome="win",
        objective_completed=True,
        learner_score=3,
        opponent_score=0,
        decisions=100,
        attack_decisions=4 if aggressive else 1,
        valid_hits=2 if aggressive else 0,
        engagement_initiations=2 if aggressive else 0,
        forward_hits=1 if aggressive else 0,
        invalid_attack_decisions=2 if aggressive else 1,
        objective_chase_decisions=0,
        retreat_decisions=0,
        terminated=True,
        truncated=False,
        peer_tic_lag_max=0,
        protocol_inconsistent=False,
        action_tic_inconsistent=False,
        score_event_inconsistent=False,
        scenario_hash="scenario",
        environment_attempts=1,
    )


def _complete_records() -> list[StyleEpisodeRecord]:
    records = []
    for pair_index, opponent in enumerate(DUEL_OPPONENTS):
        for policy in STYLE_POLICIES:
            for side in ("host", "opponent"):
                records.append(
                    _record(
                        policy,
                        opponent,
                        side,
                        pair_index=pair_index,
                    )
                )
    return records


def test_aggressive_summary_passes_retention_style_and_integrity_gates() -> None:
    summary = evaluate_aggressive_records(
        _complete_records(),
        expected_pairs_per_opponent=1,
        expected_scenario_hash="scenario",
        bootstrap_seed=7,
        bootstrap_samples=500,
    )

    assert summary.complete is True
    assert summary.passed is True
    assert summary.skill_retention == 1.0
    assert summary.per_opponent_retention == {
        opponent: 1.0 for opponent in DUEL_OPPONENTS
    }
    assert summary.engagement_initiation_delta == 2.0
    assert summary.engagement_initiation_delta_ci[0] > 0
    assert summary.policies["aggressive"].valid_attack_rate == 0.5


def test_blind_fire_and_objective_chase_fail_anti_hacking_gate() -> None:
    records = _complete_records()
    records = [
        replace(
            record,
            invalid_attack_decisions=record.attack_decisions,
            valid_hits=0,
            objective_chase_decisions=record.attack_decisions,
        )
        if record.policy == "aggressive"
        else record
        for record in records
    ]

    summary = evaluate_aggressive_records(
        records,
        expected_pairs_per_opponent=1,
        expected_scenario_hash="scenario",
        bootstrap_seed=7,
        bootstrap_samples=100,
    )

    assert summary.gates["valid_attack_rate"] is False
    assert summary.gates["objective_chase_controlled"] is False
    assert summary.passed is False


def test_missing_side_swap_or_duplicate_row_fails_completeness() -> None:
    records = _complete_records()
    records[-1] = records[-2]

    summary = evaluate_aggressive_records(
        records,
        expected_pairs_per_opponent=1,
        expected_scenario_hash="scenario",
        bootstrap_seed=7,
        bootstrap_samples=100,
    )

    assert summary.complete is False
    assert summary.gates["complete"] is False
    assert summary.protocol_inconsistencies > 0


def test_no_attack_is_not_misreported_as_perfect_attack_quality() -> None:
    records = [
        replace(
            record,
            attack_decisions=0,
            valid_hits=0,
            forward_hits=0,
            invalid_attack_decisions=0,
            objective_chase_decisions=0,
        )
        if record.policy == "aggressive"
        else record
        for record in _complete_records()
    ]

    summary = evaluate_aggressive_records(
        records,
        expected_pairs_per_opponent=1,
        expected_scenario_hash="scenario",
        bootstrap_seed=7,
        bootstrap_samples=100,
    )

    aggressive = summary.policies["aggressive"]
    assert aggressive.valid_attack_rate == 0.0
    assert aggressive.invalid_attack_rate == 0.0
    assert aggressive.objective_chase_rate == 0.0
    assert summary.gates["valid_attack_rate"] is False
