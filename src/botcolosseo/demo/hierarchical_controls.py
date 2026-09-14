"""Explicit public runtime control events for same-checkpoint demonstrations."""

from botcolosseo.agents.hierarchical_controller import HierarchicalController
from botcolosseo.training.hierarchical_protocol import ControlCondition


def control_schedule(mode="style", *, difficulty=1.0):
    """Predeclared independent or joint controls; no state-dependent override."""
    if mode not in ("style", "difficulty", "joint"):
        raise ValueError("Unknown control schedule mode")
    styles = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    levels = [difficulty] * 4 if mode == "style" else [difficulty, 0.0, 0.5, 1.0]
    if mode == "difficulty":
        styles = [(0, 0, 0)] * 4
    return [
        (time, ControlCondition(*style, level))
        for time, style, level in zip((0, 81, 161, 241), styles, levels, strict=True)
    ]


class ScheduledController(HierarchicalController):
    def __init__(self, executor, strategy, *, seed, schedule):
        if not schedule or schedule[0][0] != 0:
            raise ValueError("Control schedule must begin at decision zero")
        times = [time for time, _ in schedule]
        if any(type(time) is not int or time < 0 for time in times) or times != sorted(set(times)):
            raise ValueError("Control event decisions must strictly increase")
        if any(not isinstance(condition, ControlCondition) for _, condition in schedule):
            raise ValueError("Control schedule needs validated conditions")
        self.schedule = tuple(schedule)
        super().__init__(executor, strategy, seed=seed)

    def resolve_condition(self, condition):
        requested = self.schedule[0][1]
        for start, value in self.schedule:
            if start > self.decisions:
                break
            requested = value
        return requested

    def step(self, frames, scalars, previous_actions, condition, *, public_event=False):
        requested = self.resolve_condition(condition)
        result = super().step(
            frames, scalars, previous_actions, requested, public_event=public_event
        )
        result["requested_style"] = requested.as_tuple()[:3]
        result["requested_difficulty"] = requested.difficulty
        return result
