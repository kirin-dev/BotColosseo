from __future__ import annotations

import pytest

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_protocol import DuelEvent, DuelEventType
from botcolosseo.envs.duel_types import DuelPrivilegedState
from botcolosseo.training.defensive_reward import (
    DefensiveRewardConfig,
    DefensiveRewardLedger,
)


def _state(**overrides: object) -> DuelPrivilegedState:
    values: dict[str, object] = {
        "host_x": 0.0,
        "host_y": 0.0,
        "host_angle": 0.0,
        "host_region": "middle",
        "opponent_x": 640.0,
        "opponent_y": 0.0,
        "opponent_angle": 180.0,
        "opponent_region": "away",
        "core_x": 0.0,
        "core_y": 0.0,
        "carrier": 0,
        "host_health": 100.0,
        "opponent_health": 100.0,
        "host_score": 0,
        "opponent_score": 0,
        "round_state": 1,
        "engine_tic": 0,
    }
    values.update(overrides)
    return DuelPrivilegedState(**values)  # type: ignore[arg-type]


def _event(kind: DuelEventType, side: str) -> DuelEvent:
    return DuelEvent(kind, side, 0, 0, 0)


def test_defensive_reward_encourages_presence_entry_and_resolution() -> None:
    ledger = DefensiveRewardLedger(DefensiveRewardConfig(), learner_side="host")
    before = _state(host_x=0.0, carrier=2)
    after = _state(host_x=-640.0, carrier=0, core_x=0.0)

    reward = ledger.apply(
        MacroAction.MOVE_BACKWARD,
        (),
        has_core=False,
        state_before=before,
        state_after=after,
    )

    assert reward.components == {
        "risk_presence": pytest.approx(0.01),
        "protective_entry": pytest.approx(0.05),
        "risk_resolution": pytest.approx(0.10),
    }


def test_defensive_reward_counts_denial_and_defensive_recovery() -> None:
    ledger = DefensiveRewardLedger(DefensiveRewardConfig(), learner_side="host")
    denial = ledger.apply(
        MacroAction.ATTACK,
        (_event(DuelEventType.VALID_HIT, "host"),),
        has_core=False,
        state_before=_state(carrier=2),
        state_after=_state(carrier=0),
    )
    recovery = ledger.apply(
        MacroAction.MOVE_FORWARD,
        (),
        has_core=False,
        state_before=_state(core_x=-300.0, carrier=0),
        state_after=_state(core_x=-300.0, carrier=1),
    )

    assert denial.components["carrier_denial"] == pytest.approx(0.20)
    assert recovery.components["defensive_recovery"] == pytest.approx(0.15)


def test_defensive_reward_penalizes_guarding_and_concession() -> None:
    ledger = DefensiveRewardLedger(DefensiveRewardConfig(), learner_side="host")
    guarding = ledger.apply(
        MacroAction.IDLE,
        (),
        has_core=False,
        state_before=_state(host_x=-640.0),
        state_after=_state(host_x=-640.0),
    )
    concession = ledger.apply(
        MacroAction.IDLE,
        (_event(DuelEventType.SCORE, "opponent"),),
        has_core=False,
        state_before=_state(host_x=-640.0, carrier=2),
        state_after=_state(host_x=-640.0),
    )

    assert guarding.components == {"unnecessary_guard": pytest.approx(-0.02)}
    assert concession.components["risk_concession"] == pytest.approx(-0.20)


def test_defensive_reward_is_side_mirrored() -> None:
    ledger = DefensiveRewardLedger(DefensiveRewardConfig(), learner_side="opponent")

    reward = ledger.apply(
        MacroAction.MOVE_BACKWARD,
        (),
        has_core=False,
        state_before=_state(opponent_x=0.0, carrier=1),
        state_after=_state(opponent_x=640.0, carrier=0),
    )

    assert reward.components["protective_entry"] > 0
    assert reward.components["risk_resolution"] > 0


def test_defensive_reward_caps_threat_prolongation_and_resets() -> None:
    config = DefensiveRewardConfig(risk_presence_cap=2)
    ledger = DefensiveRewardLedger(config, learner_side="host")
    state = _state(host_x=-640.0, carrier=2)

    totals = [
        ledger.apply(
            MacroAction.IDLE,
            (),
            has_core=False,
            state_before=state,
            state_after=state,
        ).total
        for _ in range(3)
    ]
    ledger.reset()
    reset = ledger.apply(
        MacroAction.IDLE,
        (),
        has_core=False,
        state_before=state,
        state_after=state,
    )

    assert totals == pytest.approx([0.01, 0.01, 0.0])
    assert reset.total == pytest.approx(0.01)


def test_defensive_reward_requires_privileged_training_state() -> None:
    ledger = DefensiveRewardLedger(DefensiveRewardConfig(), learner_side="host")

    with pytest.raises(ValueError, match="before/after"):
        ledger.apply(MacroAction.IDLE, (), has_core=False)
