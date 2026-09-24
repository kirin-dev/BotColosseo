import torch

from botcolosseo.agents.hierarchical_model import CommandExecutor, StrategicActor
from botcolosseo.demo.hierarchical_controls import ScheduledController, control_schedule
from botcolosseo.training.hierarchical_protocol import ControlCondition


def test_counterbalanced_order_preserves_times_and_hard_difficulty():
    from itertools import permutations

    for order in permutations(("aggressive", "defensive", "explorer")):
        schedule = control_schedule(style_order=order)
        assert [t for t, _ in schedule] == [0, 81, 161, 241]
        assert all(c.difficulty == 1 for _, c in schedule)
        assert [getattr(c, name) for (_, c), name in
                zip(schedule[1:], order, strict=True)] == [1, 1, 1]


def test_runtime_schedule_retains_memory_and_respects_boundaries():
    torch.set_num_threads(1)
    neutral = ControlCondition()
    changed = ControlCondition(aggressive=1, difficulty=0.5)
    controller = ScheduledController(
        CommandExecutor(), StrategicActor(), seed=17, schedule=[(0, neutral), (3, changed)]
    )
    inputs = (
        torch.zeros(1, 1, 1, 84, 84, dtype=torch.uint8),
        torch.zeros(1, 1, 9),
        torch.zeros(1, 1, dtype=torch.long),
    )
    records = [controller.step(*inputs, neutral) for _ in range(3)]
    high = controller.high_hidden
    records.append(controller.step(*inputs, neutral))
    assert controller.high_hidden is high
    assert records[-1]["requested_style"] == (1, 0, 0)
    assert records[-1]["style"] == (0, 0, 0)
    assert records[-1]["difficulty"] == 0.5
    assert controller.applied_low_difficulty == 0.5
    assert controller.requested_condition == changed
    assert controller.applied_condition == neutral
    for _ in range(5):
        records.append(controller.step(*inputs, neutral))
    assert records[-1]["replanned"] and records[-1]["style"] == (1, 0, 0)
    assert controller.decisions == 9 and controller.low_hidden is not None


def test_control_schedules_isolate_axes_and_preserve_legacy_style_track():
    style = control_schedule()
    assert [t for t, _ in style] == [0, 81, 161, 241]
    assert [c.as_tuple() for _, c in style] == [
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 1, 0, 1),
        (0, 0, 1, 1),
    ]
    difficulty = control_schedule("difficulty")
    assert all(c.as_tuple()[:3] == (0, 0, 0) for _, c in difficulty)
    assert [c.difficulty for _, c in difficulty] == [1, 0, 0.5, 1]
    joint = control_schedule("joint")
    assert joint[1][1].as_tuple() == (1, 0, 0, 0)
    assert joint[2][1].as_tuple() == (0, 1, 0, 0.5)


def test_integer_runtime_conditions_are_converted_to_float_model_inputs():
    torch.set_num_threads(1)
    controller = ScheduledController(
        CommandExecutor(), StrategicActor(), seed=17, schedule=control_schedule("joint")
    )
    inputs = (
        torch.zeros(1, 1, 1, 84, 84, dtype=torch.uint8),
        torch.zeros(1, 1, 9),
        torch.zeros(1, 1, dtype=torch.long),
    )
    controller.step(*inputs, ControlCondition())
    assert controller.last_high_inputs["style"].dtype == torch.float32
    assert controller.last_high_inputs["difficulty"].dtype == torch.float32
