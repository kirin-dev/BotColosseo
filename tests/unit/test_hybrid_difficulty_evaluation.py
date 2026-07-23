from __future__ import annotations

from botcolosseo.agents.difficulty import DifficultyProfile
from botcolosseo.agents.hybrid_difficulty import HybridExecutionTrace
from botcolosseo.agents.style_governor import GovernorTelemetry
from botcolosseo.envs.actions import MacroAction
from botcolosseo.evaluation.defensive import DefensiveEpisodeRecord
from botcolosseo.evaluation.hybrid_difficulty import (
    HybridDifficultyExecutionRow,
    HybridDifficultyGovernorRow,
    evaluate_hybrid_difficulty_extension,
)
from botcolosseo.evaluation.native_style_difficulty import (
    NativeStyleDifficultyRecord,
)
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS

SCENARIO = "scenario"
PROFILES = {
    "easy": DifficultyProfile(2, 2),
    "normal": DifficultyProfile(1, 1),
}


def _episode(difficulty: str, opponent: str, side: str) -> NativeStyleDifficultyRecord:
    return NativeStyleDifficultyRecord(
        difficulty,
        DefensiveEpisodeRecord(
            policy="defensive",
            split="validation",
            opponent=opponent,
            pair_index=0,
            seed=1,
            learner_side=side,
            outcome="win",
            objective_completed=True,
            learner_score=1,
            opponent_score=0,
            decisions=8,
            risk_decisions=4,
            protective_presence_decisions=4,
            carrier_opportunities=0,
            carrier_denials=0,
            recovery_opportunities=0,
            recoveries=0,
            no_risk_decisions=4,
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
        ),
    )


def _evidence(records: list[NativeStyleDifficultyRecord]):
    governors = []
    executions = []
    states = ("guard", "disengage", "recover")
    for record in records:
        profile = PROFILES[record.difficulty]
        updates = 4 if record.difficulty == "easy" else 8
        telemetry = [
            GovernorTelemetry(
                decision_index=index,
                base_action=MacroAction.IDLE,
                final_action=MacroAction.MOVE_FORWARD,
                state=states[index % len(states)],
                trigger="trigger",
                reason="reason",
                intervened=True,
                used_override=False,
                fallback_condition="timeout",
                route_mode=None,
            )
            for index in range(updates)
        ]
        governors.extend(
            HybridDifficultyGovernorRow.from_telemetry(
                style="defensive",
                difficulty=record.difficulty,
                opponent=record.episode.opponent,
                pair_index=record.episode.pair_index,
                learner_side=record.episode.learner_side,
                telemetry=row,
            )
            for row in telemetry
        )
        delay = profile.reaction_delay
        for index in range(record.episode.decisions):
            warmup = index < delay
            source = None if warmup else (index - delay) // profile.policy_update_interval
            trace = HybridExecutionTrace(
                decision_index=index,
                policy_updated=index % profile.policy_update_interval == 0,
                proposed_action=MacroAction.MOVE_FORWARD,
                emitted_action=(
                    MacroAction.IDLE if warmup else MacroAction.MOVE_FORWARD
                ),
                source_decision_index=source,
                base_action=None if warmup else MacroAction.IDLE,
                state="warmup" if warmup else states[source % len(states)],
                trigger="warmup" if warmup else "trigger",
                reason="fifo_warmup" if warmup else "reason",
                intervened=not warmup,
                used_override=False,
                fallback_condition="warmup" if warmup else "timeout",
                route_mode=None,
                warmup=warmup,
            )
            executions.append(
                HybridDifficultyExecutionRow.from_trace(
                    style="defensive",
                    difficulty=record.difficulty,
                    opponent=record.episode.opponent,
                    pair_index=record.episode.pair_index,
                    learner_side=record.episode.learner_side,
                    trace=trace,
                )
            )
    return governors, executions


def test_hybrid_difficulty_extension_accepts_complete_executed_evidence() -> None:
    records = [
        _episode(difficulty, opponent, side)
        for difficulty in ("easy", "normal")
        for opponent in DUEL_OPPONENTS
        for side in ("host", "opponent")
    ]
    governors, executions = _evidence(records)

    result = evaluate_hybrid_difficulty_extension(
        records,
        governors,
        executions,
        style="defensive",
        profiles=PROFILES,
        max_consecutive_interventions=12,
        expected_pairs_per_opponent=1,
        expected_scenario_hash=SCENARIO,
    )

    assert result["passed"] is True
    assert result["episodes"] == 20
    assert result["governor_decisions"] == 120
    assert result["executed_decisions"] == 160
    assert all(tier["passed"] for tier in result["tiers"].values())


def test_hybrid_difficulty_extension_rejects_missing_execution_trace() -> None:
    records = [
        _episode(difficulty, opponent, side)
        for difficulty in ("easy", "normal")
        for opponent in DUEL_OPPONENTS
        for side in ("host", "opponent")
    ]
    governors, executions = _evidence(records)

    result = evaluate_hybrid_difficulty_extension(
        records,
        governors,
        executions[:-1],
        style="defensive",
        profiles=PROFILES,
        max_consecutive_interventions=12,
        expected_pairs_per_opponent=1,
        expected_scenario_hash=SCENARIO,
    )

    assert result["passed"] is False
    assert result["tiers"]["normal"]["gates"]["evidence_complete"] is False
