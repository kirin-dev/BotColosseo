from pathlib import Path

import pytest

from botcolosseo.envs.duel_protocol import DuelEvent, DuelEventType
from botcolosseo.envs.duel_rewards import DuelRewardLedger, load_reward_config


def event(event_type: DuelEventType, side: str) -> DuelEvent:
    return DuelEvent(event_type, side, episode_id=0, decision_index=1, engine_tic=4)


@pytest.mark.parametrize(
    "event_type",
    (DuelEventType.SCORE, DuelEventType.VALID_HIT, DuelEventType.DEATH),
)
def test_competitive_event_rewards_are_zero_sum(event_type: DuelEventType) -> None:
    ledger = DuelRewardLedger(load_reward_config(Path("configs/m2/reward.yaml")))

    rewards = ledger.apply((event(event_type, "host"),))

    assert rewards.host == pytest.approx(-rewards.opponent)
    assert rewards.host != 0


def test_event_caps_prevent_repeated_shaping_farms() -> None:
    ledger = DuelRewardLedger(load_reward_config(Path("configs/m2/reward.yaml")))
    pickups = tuple(event(DuelEventType.PICKUP, "host") for _ in range(10))

    first = ledger.apply(pickups)
    second = ledger.apply(pickups)

    assert first.host == pytest.approx(0.1)
    assert second.host == 0.0
    assert first.opponent == pytest.approx(-0.1)


def test_reset_restores_caps() -> None:
    ledger = DuelRewardLedger(load_reward_config(Path("configs/m2/reward.yaml")))
    pickup = (event(DuelEventType.PICKUP, "host"),)
    ledger.apply(pickup)

    ledger.reset()

    assert ledger.apply(pickup).host == pytest.approx(0.1)
