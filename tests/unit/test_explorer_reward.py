from __future__ import annotations

import pytest

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_protocol import DuelEvent, DuelEventType
from botcolosseo.envs.duel_types import DuelPrivilegedState
from botcolosseo.training.explorer_reward import (
    ExplorerRewardConfig,
    ExplorerRewardLedger,
)


def _state(**overrides: object) -> DuelPrivilegedState:
    values: dict[str, object] = {
        "host_x": 0.0,
        "host_y": 0.0,
        "host_angle": 0.0,
        "host_region": "middle",
        "opponent_x": 0.0,
        "opponent_y": 0.0,
        "opponent_angle": 180.0,
        "opponent_region": "middle",
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


def _score(side: str = "host") -> tuple[DuelEvent, ...]:
    return (DuelEvent(DuelEventType.SCORE, side, 0, 0, 0),)


def _apply(
    ledger: ExplorerRewardLedger,
    before: DuelPrivilegedState,
    after: DuelPrivilegedState,
    events: tuple[DuelEvent, ...] = (),
):
    return ledger.apply(
        MacroAction.MOVE_FORWARD,
        events,
        has_core=before.carrier == 1,
        state_before=before,
        state_after=after,
    )


def test_explorer_reward_cycles_upper_lower_and_flank_targets() -> None:
    ledger = ExplorerRewardLedger(ExplorerRewardConfig(), learner_side="host")
    upper = _apply(
        ledger,
        _state(carrier=0),
        _state(carrier=1, host_region="upper_route"),
    )
    upper_score = _apply(
        ledger,
        _state(carrier=1, host_region="upper_route"),
        _state(carrier=0, host_score=1),
        _score(),
    )
    lower = _apply(
        ledger,
        _state(carrier=0, host_score=1),
        _state(carrier=1, host_score=1, host_region="lower_route"),
    )
    lower_score = _apply(
        ledger,
        _state(carrier=1, host_score=1, host_region="lower_route"),
        _state(carrier=0, host_score=2),
        _score(),
    )
    west = _apply(
        ledger,
        _state(carrier=0, host_score=2),
        _state(carrier=1, host_score=2, host_region="flank_west"),
    )
    east = _apply(
        ledger,
        _state(carrier=1, host_score=2, host_region="flank_west"),
        _state(carrier=1, host_score=2, host_region="flank_east"),
    )
    flank_score = _apply(
        ledger,
        _state(carrier=1, host_score=2, host_region="flank_east"),
        _state(carrier=0, host_score=3),
        _score(),
    )

    assert upper.components["target_region"] > 0
    assert upper_score.components["target_route_score"] > 0
    assert lower.components["target_region"] > 0
    assert lower_score.components["target_route_score"] > 0
    assert west.components["target_region"] > 0
    assert east.components["target_region"] > 0
    assert flank_score.components["target_route_score"] > 0


def test_explorer_reward_does_not_reward_noncarry_or_wrong_route_score() -> None:
    ledger = ExplorerRewardLedger(ExplorerRewardConfig(), learner_side="host")
    noncarry = _apply(
        ledger,
        _state(),
        _state(host_region="upper_route"),
    )
    _apply(
        ledger,
        _state(),
        _state(carrier=1, host_region="lower_route"),
    )
    wrong_score = _apply(
        ledger,
        _state(carrier=1, host_region="lower_route"),
        _state(carrier=0, host_score=1),
        _score(),
    )

    assert noncarry.total == 0
    assert "target_route_score" not in wrong_score.components


def test_explorer_reward_penalizes_carry_stall_with_cap_and_reset() -> None:
    config = ExplorerRewardConfig(stall_decisions=1, carry_stall_cap=2)
    ledger = ExplorerRewardLedger(config, learner_side="host")
    state = _state(carrier=1, host_region="middle")

    totals = [
        _apply(ledger, state, state).components.get("carry_stall", 0.0)
        for _ in range(4)
    ]
    ledger.reset()
    reset = _apply(ledger, state, state)

    assert totals == pytest.approx([0.0, -0.01, -0.01, 0.0])
    assert "carry_stall" not in reset.components


def test_explorer_reward_uses_opponent_side_region_and_score() -> None:
    ledger = ExplorerRewardLedger(ExplorerRewardConfig(), learner_side="opponent")
    reward = _apply(
        ledger,
        _state(carrier=0),
        _state(carrier=2, opponent_region="upper_route"),
    )

    assert reward.components["target_region"] > 0


def test_explorer_reward_requires_privileged_training_state() -> None:
    ledger = ExplorerRewardLedger(ExplorerRewardConfig(), learner_side="host")

    with pytest.raises(ValueError, match="before/after"):
        ledger.apply(MacroAction.IDLE, (), has_core=False)
