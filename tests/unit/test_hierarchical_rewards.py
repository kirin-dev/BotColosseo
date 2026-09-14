import pytest

from botcolosseo.training.hierarchical_rewards import command_reward


def reward(previous, following, *, terminal=False):
    return command_reward(
        previous_banked=0,
        banked=0,
        previous_potential=previous,
        next_potential=following,
        gamma=0.9,
        terminated=terminal,
        newly_completed=False,
    )


def test_pbrs_telescopes_across_augmented_command_states():
    # Middle value can change because the command changed; no ad hoc reset.
    potentials = [0.2, -0.5, 0.7, 0.4]
    total = sum(
        0.9**i * reward(potentials[i], potentials[i + 1], terminal=i == 2).potential
        for i in range(3)
    )
    assert total == pytest.approx(-potentials[0])


def test_truncation_keeps_potential():
    assert reward(0.2, 0.5).potential == pytest.approx(0.25)
    assert reward(0.2, 0.5, terminal=True).potential == pytest.approx(-0.2)


def test_own_payoff_and_skill_bonus_are_separate():
    result = command_reward(
        previous_banked=0,
        banked=75,
        previous_potential=0,
        next_potential=0,
        gamma=0.99,
        terminated=True,
        newly_completed=True,
    )
    assert result.task == 0.5
    assert result.command_success == 0.05


def test_low_level_task_weight_does_not_change_skill_bonus_or_potential():
    kwargs = dict(
        previous_banked=0,
        banked=75,
        previous_potential=-0.5,
        next_potential=0,
        gamma=0.997,
        terminated=True,
        newly_completed=True,
    )
    original = command_reward(**kwargs)
    calibrated = command_reward(**kwargs, task_weight=0.1)
    assert calibrated.task == pytest.approx(0.05)
    assert calibrated.command_success == original.command_success
    assert calibrated.potential == original.potential
    for invalid in (0, -1, 2, float("nan")):
        with pytest.raises(ValueError):
            command_reward(**kwargs, task_weight=invalid)
