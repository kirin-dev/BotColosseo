import pytest

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_protocol import DuelEvent, DuelEventType
from botcolosseo.training.aggressive_reward import (
    AggressiveRewardConfig,
    AggressiveRewardLedger,
)


def _hit(index: int = 0) -> DuelEvent:
    return DuelEvent(DuelEventType.VALID_HIT, "host", 0, index, index)


def test_valid_engagement_is_rewarded_but_blind_fire_is_penalized() -> None:
    ledger = AggressiveRewardLedger(AggressiveRewardConfig(), learner_side="host")

    hit = ledger.apply(MacroAction.FORWARD_ATTACK, (_hit(),), has_core=False)
    miss = ledger.apply(MacroAction.ATTACK, (), has_core=False)

    assert hit.total > 0
    assert hit.components["valid_hit"] > 0
    assert hit.components["engagement_initiation"] > 0
    assert miss.total < 0
    assert miss.components == {"invalid_attack": pytest.approx(-0.02)}


def test_opponent_hit_and_non_attack_never_create_aggression_reward() -> None:
    ledger = AggressiveRewardLedger(AggressiveRewardConfig(), learner_side="host")
    opponent_hit = DuelEvent(DuelEventType.VALID_HIT, "opponent", 0, 0, 0)

    assert ledger.apply(MacroAction.MOVE_FORWARD, (opponent_hit,), has_core=False).total == 0
    assert ledger.apply(MacroAction.IDLE, (), has_core=False).total == 0


def test_reward_caps_and_core_chase_guard_prevent_hacking() -> None:
    config = AggressiveRewardConfig(valid_hit_cap=1, initiation_cap=1)
    ledger = AggressiveRewardLedger(config, learner_side="host")

    first = ledger.apply(MacroAction.ATTACK, (_hit(0),), has_core=False)
    second = ledger.apply(MacroAction.ATTACK, (_hit(1),), has_core=False)
    carrying = ledger.apply(MacroAction.ATTACK, (), has_core=True)

    assert first.total > 0
    assert second.total == 0
    assert carrying.components["invalid_attack"] < 0
    assert carrying.components["objective_chase"] < 0
