from dataclasses import replace

import pytest

from botcolosseo.training.hierarchical_protocol import (
    Command,
    ControlCondition,
    GameIdentity,
    own_payoffs,
    should_replan,
    smdp_target,
)


def test_command_schema_and_replanning_boundaries():
    assert len(Command) == 7
    assert not should_replan(1, public_event=True)
    assert should_replan(2, public_event=True)
    assert not should_replan(7)
    assert should_replan(8)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_control_rejected(value):
    with pytest.raises(ValueError):
        ControlCondition(difficulty=value)


def test_smdp_terminal_and_truncation():
    assert smdp_target((1, 2), 0.5, 8, terminated=True) == 2
    assert smdp_target((1, 2), 0.5, 8, terminated=False) == 4


def test_general_sum_payoff_and_global_settlement():
    assert own_payoffs(150, 150, globally_settled=True) == (1, 1)
    assert own_payoffs(30, 90, globally_settled=True) == (0.2, 0.6)
    with pytest.raises(ValueError):
        own_payoffs(150, 0, globally_settled=False)
    with pytest.raises(ValueError):
        own_payoffs(151, 0, globally_settled=True)


def test_executor_and_conditions_invalidate_payoffs():
    game = GameIdentity("scenario", "rules", "executor1", ControlCondition(), ControlCondition())
    game.require_same_game(replace(game))
    for changed in (
        replace(game, executor_hash="executor2"),
        replace(game, host_condition=ControlCondition(aggressive=1)),
        replace(game, inference="low-sample"),
    ):
        with pytest.raises(ValueError, match="Stale"):
            game.require_same_game(changed)
