from __future__ import annotations

from dataclasses import replace

from botcolosseo.envs.actions import MacroAction
from botcolosseo.evaluation.defensive import (
    DefensiveEpisodeRecord,
    DefensiveEvaluationSummary,
    DefensivePolicySummary,
)
from botcolosseo.evaluation.hybrid import (
    HybridTelemetryRow,
    evaluate_hybrid_product,
)


def _record(policy: str, *, decisions: int = 3) -> DefensiveEpisodeRecord:
    return DefensiveEpisodeRecord(
        policy=policy,
        split="validation",
        opponent="random_legal",
        pair_index=0,
        seed=1,
        learner_side="host",
        outcome="win",
        objective_completed=True,
        learner_score=1,
        opponent_score=0,
        decisions=decisions,
        risk_decisions=0,
        protective_presence_decisions=0,
        carrier_opportunities=0,
        carrier_denials=0,
        recovery_opportunities=0,
        recoveries=0,
        no_risk_decisions=decisions,
        unnecessary_guard_decisions=0,
        low_health_opportunities=0,
        successful_escapes=0,
        terminated=True,
        truncated=False,
        peer_tic_lag_max=0,
        protocol_inconsistent=False,
        action_tic_inconsistent=False,
        score_event_inconsistent=False,
        scenario_hash="scenario",
    )


def _policy_summary() -> DefensivePolicySummary:
    return DefensivePolicySummary(
        episodes=1,
        win_rate=1.0,
        objective_rate=1.0,
        mean_score_difference=1.0,
        performance=1.0,
        risk_decisions=0,
        protective_presence_rate=0.0,
        carrier_opportunities=0,
        carrier_denials=0,
        recovery_opportunities=0,
        recoveries=0,
        denial_recovery_rate=0.0,
        unnecessary_guard_rate=0.0,
        low_health_opportunities=0,
        successful_escapes=0,
    )


def _legacy(*, retention: float = 0.9) -> DefensiveEvaluationSummary:
    return DefensiveEvaluationSummary(
        complete=True,
        passed=False,
        episodes=2,
        expected_episodes=2,
        protocol_inconsistencies=0,
        environment_retries=0,
        skill_retention=retention,
        per_opponent_retention={"random_legal": retention},
        protective_presence_estimator="frozen",
        protective_presence_delta=-0.1,
        protective_presence_delta_ci=(-0.2, 0.1),
        gates={"legacy_style_metric": False},
        policies={"strong_base": _policy_summary(), "defensive": _policy_summary()},
    )


def _telemetry() -> list[HybridTelemetryRow]:
    return [
        HybridTelemetryRow(
            style="defensive",
            opponent="random_legal",
            pair_index=0,
            learner_side="host",
            decision_index=index,
            base_action=MacroAction.MOVE_FORWARD,
            final_action=(
                MacroAction.STRAFE_LEFT if index < 2 else MacroAction.MOVE_FORWARD
            ),
            state=("guard", "disengage", "recover")[index],
            trigger=("own_score_rise", "health_drop", "recover_cooldown")[index],
            reason=("guard_scan_left", "disengage_left", "exact_base_fallback")[index],
            intervened=index < 2,
            used_override=False,
            fallback_condition="bounded",
            route_mode=None,
        )
        for index in range(3)
    ]


def test_product_gate_can_pass_while_frozen_legacy_style_gate_remains_negative() -> None:
    records = [_record("strong_base"), _record("defensive")]

    summary = evaluate_hybrid_product(
        style="defensive",
        records=records,
        telemetry=_telemetry(),
        legacy_summary=_legacy(),
        max_consecutive_interventions=2,
    )

    assert summary.passed
    assert not summary.legacy_diagnostic_passed
    assert summary.interventions == 2
    assert summary.max_consecutive_interventions == 2
    assert all(summary.gates.values())


def test_product_gate_fails_closed_on_retention_or_incomplete_telemetry() -> None:
    records = [_record("strong_base"), _record("defensive")]

    low_retention = evaluate_hybrid_product(
        style="defensive",
        records=records,
        telemetry=_telemetry(),
        legacy_summary=_legacy(retention=0.7),
        max_consecutive_interventions=2,
    )
    missing = evaluate_hybrid_product(
        style="defensive",
        records=records,
        telemetry=_telemetry()[:-1],
        legacy_summary=_legacy(),
        max_consecutive_interventions=2,
    )

    assert not low_retention.passed
    assert not low_retention.gates["skill_retention"]
    assert not low_retention.gates["per_opponent_retention"]
    assert not missing.passed
    assert not missing.gates["telemetry_complete"]


def test_product_gate_rejects_unbounded_consecutive_intervention() -> None:
    records = [_record("strong_base"), _record("defensive")]
    rows = [replace(row, intervened=True) for row in _telemetry()]

    summary = evaluate_hybrid_product(
        style="defensive",
        records=records,
        telemetry=rows,
        legacy_summary=_legacy(),
        max_consecutive_interventions=2,
    )

    assert not summary.passed
    assert not summary.gates["intervention_bounded"]
