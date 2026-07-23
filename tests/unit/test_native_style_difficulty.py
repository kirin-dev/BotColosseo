from __future__ import annotations

import pytest

from botcolosseo.evaluation.defensive import DefensiveEpisodeRecord
from botcolosseo.evaluation.explorer import ExplorerEpisodeRecord
from botcolosseo.evaluation.native_style_difficulty import (
    NativeStyleDifficultyRecord,
    evaluate_native_style_difficulty,
)
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS

SCENARIO = "scenario"


def _defensive(
    policy: str,
    opponent: str,
    pair_index: int,
    side: str,
) -> DefensiveEpisodeRecord:
    styled = policy == "defensive"
    return DefensiveEpisodeRecord(
        policy=policy,
        split="validation",
        opponent=opponent,
        pair_index=pair_index,
        seed=100 + pair_index,
        learner_side=side,
        outcome="win",
        objective_completed=True,
        learner_score=1,
        opponent_score=0,
        decisions=20,
        risk_decisions=10,
        protective_presence_decisions=10 if styled else 0,
        carrier_opportunities=1,
        carrier_denials=1 if styled else 0,
        recovery_opportunities=1,
        recoveries=1 if styled else 0,
        no_risk_decisions=10,
        unnecessary_guard_decisions=0,
        low_health_opportunities=0,
        successful_escapes=0,
        terminated=True,
        truncated=False,
        peer_tic_lag_max=0,
        protocol_inconsistent=False,
        action_tic_inconsistent=False,
        score_event_inconsistent=False,
        scenario_hash=SCENARIO,
    )


def _explorer(
    policy: str,
    opponent: str,
    pair_index: int,
    side: str,
) -> ExplorerEpisodeRecord:
    styled = policy == "explorer"
    return ExplorerEpisodeRecord(
        policy=policy,
        split="validation",
        opponent=opponent,
        pair_index=pair_index,
        seed=100 + pair_index,
        learner_side=side,
        outcome="win",
        objective_completed=True,
        learner_score=3,
        opponent_score=0,
        learner_scores=3,
        decisions=30,
        upper_completions=1 if styled else 3,
        lower_completions=1 if styled else 0,
        flank_completions=1 if styled else 0,
        mixed_or_unknown_completions=0,
        unique_regions=6 if styled else 3,
        terminated=True,
        truncated=False,
        peer_tic_lag_max=0,
        protocol_inconsistent=False,
        action_tic_inconsistent=False,
        score_event_inconsistent=False,
        scenario_hash=SCENARIO,
    )


def _records(style: str) -> list[NativeStyleDifficultyRecord]:
    policies = ("strong_base", style)
    factory = _defensive if style == "defensive" else _explorer
    return [
        NativeStyleDifficultyRecord(
            difficulty,
            factory(policy, opponent, pair_index, side),
        )
        for difficulty in ("easy", "normal", "hard")
        for policy in policies
        for opponent in DUEL_OPPONENTS
        for pair_index, side in ((0, "host"), (0, "opponent"))
    ]


def test_defensive_difficulty_requires_native_style_gate_at_every_tier() -> None:
    result = evaluate_native_style_difficulty(
        _records("defensive"),
        style="defensive",
        expected_pairs_per_opponent=1,
        expected_scenario_hash=SCENARIO,
        bootstrap_seed=7,
        bootstrap_samples=100,
    )

    assert result["passed"] is True
    assert result["episodes"] == 60
    assert result["gates"]["style_preserved_at_every_tier"] is True
    assert result["monotonic_opponents"] == {
        "strong_base": 5,
        "defensive": 5,
    }


def test_explorer_difficulty_requires_route_gate_at_every_tier() -> None:
    result = evaluate_native_style_difficulty(
        _records("explorer"),
        style="explorer",
        expected_pairs_per_opponent=1,
        expected_scenario_hash=SCENARIO,
        bootstrap_seed=7,
        bootstrap_samples=100,
    )

    assert result["passed"] is True
    assert result["tiers"]["hard"]["route_entropy_delta"] == pytest.approx(1)
    assert result["gates"]["style_preserved_at_every_tier"] is True


def test_native_difficulty_serialization_is_style_bound() -> None:
    record = _records("defensive")[0]

    restored = NativeStyleDifficultyRecord.from_dict(
        record.to_dict(),
        style="defensive",
    )

    assert restored == record
