from __future__ import annotations

from dataclasses import replace

from botcolosseo.agents.difficulty import DIFFICULTIES
from botcolosseo.evaluation.difficulty import (
    DifficultyEpisodeRecord,
    evaluate_difficulty_records,
)
from botcolosseo.evaluation.style import STYLE_POLICIES, StyleEpisodeRecord
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS


def _episode(
    *,
    policy: str,
    opponent: str,
    pair_index: int,
    side: str,
    difficulty: str,
) -> DifficultyEpisodeRecord:
    scores = {
        "easy": (0, 1, "loss"),
        "normal": (1, 1, "draw"),
        "hard": (2, 1, "win"),
    }
    learner_score, opponent_score, outcome = scores[difficulty]
    aggressive = policy == "aggressive"
    return DifficultyEpisodeRecord(
        difficulty=difficulty,
        episode=StyleEpisodeRecord(
            policy=policy,
            split="validation",
            opponent=opponent,
            pair_index=pair_index,
            seed=pair_index,
            learner_side=side,
            outcome=outcome,
            objective_completed=True,
            learner_score=learner_score,
            opponent_score=opponent_score,
            decisions=100,
            attack_decisions=int(aggressive),
            valid_hits=int(aggressive),
            engagement_initiations=int(aggressive),
            forward_hits=0,
            invalid_attack_decisions=0,
            objective_chase_decisions=0,
            retreat_decisions=0,
            terminated=True,
            truncated=False,
            peer_tic_lag_max=0,
            protocol_inconsistent=False,
            action_tic_inconsistent=False,
            score_event_inconsistent=False,
            scenario_hash="scenario",
        ),
    )


def _records() -> list[DifficultyEpisodeRecord]:
    return [
        _episode(
            policy=policy,
            opponent=opponent,
            pair_index=0,
            side=side,
            difficulty=difficulty,
        )
        for policy in STYLE_POLICIES
        for difficulty in DIFFICULTIES
        for opponent in DUEL_OPPONENTS
        for side in ("host", "opponent")
    ]


def test_difficulty_gate_accepts_monotonic_style_preserving_schedule() -> None:
    result = evaluate_difficulty_records(
        _records(),
        expected_pairs_per_opponent=1,
        expected_scenario_hash="scenario",
    )

    assert result.passed is True
    assert result.episodes == 60
    assert result.monotonic_opponents == {
        "strong_base": 5,
        "aggressive": 5,
    }
    assert all(result.gates.values())


def test_difficulty_gate_rejects_inverted_aggregate_performance() -> None:
    records = _records()
    records = [
        DifficultyEpisodeRecord(
            difficulty=row.difficulty,
            episode=replace(
                row.episode,
                learner_score=2,
                opponent_score=1,
                outcome="win",
            ),
        )
        if row.episode.policy == "strong_base" and row.difficulty == "easy"
        else row
        for row in records
    ]

    result = evaluate_difficulty_records(
        records,
        expected_pairs_per_opponent=1,
        expected_scenario_hash="scenario",
    )

    assert result.passed is False
    assert result.gates["aggregate_monotonic"] is False
    assert result.gates["per_opponent_monotonic"] is False


def test_difficulty_gate_rejects_style_direction_reversal() -> None:
    records = _records()
    records = [
        DifficultyEpisodeRecord(
            difficulty=row.difficulty,
            episode=replace(
                row.episode,
                attack_decisions=0,
                valid_hits=0,
                engagement_initiations=0,
            ),
        )
        if row.episode.policy == "aggressive" and row.difficulty == "easy"
        else row
        for row in records
    ]

    result = evaluate_difficulty_records(
        records,
        expected_pairs_per_opponent=1,
        expected_scenario_hash="scenario",
    )

    assert result.passed is False
    assert result.gates["style_direction_preserved"] is False


def test_difficulty_gate_detects_duplicate_and_scenario_mismatch() -> None:
    records = _records()
    records[-1] = replace(
        records[0],
        episode=replace(records[0].episode, scenario_hash="wrong"),
    )

    result = evaluate_difficulty_records(
        records,
        expected_pairs_per_opponent=1,
        expected_scenario_hash="scenario",
    )

    assert result.passed is False
    assert result.gates["complete"] is False
    assert result.gates["protocol_clean"] is False


def test_difficulty_record_round_trip_preserves_identity() -> None:
    record = _records()[0]

    loaded = DifficultyEpisodeRecord.from_dict(record.to_dict())

    assert loaded == record
    assert loaded.identity == record.identity
