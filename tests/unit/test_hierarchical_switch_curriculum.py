import pytest

from botcolosseo.training.hierarchical_protocol import ControlCondition
from botcolosseo.training.hierarchical_switch_curriculum import random_control_schedule


@pytest.mark.parametrize("mode", ["style", "difficulty", "joint"])
def test_reproducible_segments_and_independent_controls(mode):
    args = dict(seed=71, mode=mode, initial=ControlCondition(1, 0, 1, 0.5))
    schedule = random_control_schedule(**args)
    assert schedule == random_control_schedule(**args)
    assert schedule != random_control_schedule(**{**args, "seed": 72})
    assert schedule[0] == (0, args["initial"])
    for (before, a), (after, b) in zip(schedule, schedule[1:], strict=False):
        assert 40 <= after - before <= 100
        assert after < 657
        style_changed = a.as_tuple()[:3] != b.as_tuple()[:3]
        difficulty_changed = a.difficulty != b.difficulty
        assert style_changed == (mode in ("style", "joint"))
        assert difficulty_changed == (mode in ("difficulty", "joint"))


@pytest.mark.parametrize("override", [
    {"mode": "unknown"}, {"seed": -1}, {"min_segment": 0},
    {"min_segment": 101}, {"horizon": 100}, {"max_segment": 1.5},
    {"initial": (0, 0, 0, 1)},
])
def test_invalid_schedule_rejected(override):
    args = dict(seed=71, mode="style", initial=ControlCondition())
    with pytest.raises(ValueError):
        random_control_schedule(**{**args, **override})
