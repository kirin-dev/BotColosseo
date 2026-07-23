from __future__ import annotations

from dataclasses import dataclass

from botcolosseo.agents.duel_teachers import EXPLORER_ROUTE_CYCLE
from botcolosseo.envs.duel_protocol import DuelEvent, DuelEventType

_ROUTE_REGIONS = {
    "direct_upper": {"upper_route"},
    "direct_lower": {"lower_route"},
    "flank": {"flank_west", "flank_east"},
}
_FLANK_REGIONS = _ROUTE_REGIONS["flank"]


@dataclass(frozen=True)
class ExplorerStepEvidence:
    selected_route: str
    learner_region: str | None
    events: tuple[DuelEvent, ...]

    def __post_init__(self) -> None:
        if self.selected_route not in EXPLORER_ROUTE_CYCLE:
            raise ValueError("Explorer evidence route is invalid")


@dataclass(frozen=True)
class ExplorerWindowLabels:
    selected: tuple[bool, ...]
    reasons: tuple[str, ...]
    successful_windows: int
    route_windows: tuple[tuple[str, int], ...]


def label_explorer_windows(
    steps: tuple[ExplorerStepEvidence, ...], *, learner_side: str
) -> ExplorerWindowLabels:
    if learner_side not in ("host", "opponent"):
        raise ValueError("Explorer window learner side is invalid")
    opponent_side = "opponent" if learner_side == "host" else "host"
    selected = [False] * len(steps)
    reasons = ["incomplete_route_window"] * len(steps)
    route_counts = {route: 0 for route in EXPLORER_ROUTE_CYCLE}
    start = 0
    for index, step in enumerate(steps):
        learner_score = _has_event(step.events, learner_side, DuelEventType.SCORE)
        opponent_score = _has_event(step.events, opponent_side, DuelEventType.SCORE)
        learner_death = _has_event(step.events, learner_side, DuelEventType.DEATH)
        terminal = learner_score or opponent_score or learner_death or index + 1 == len(steps)
        if not terminal:
            continue
        window = steps[start : index + 1]
        routes = {item.selected_route for item in window}
        expected_route = step.selected_route
        matches = (
            len(routes) == 1
            and expected_route in routes
            and _matches_route(
                expected_route,
                {item.learner_region for item in window if item.learner_region},
            )
        )
        success = learner_score and not opponent_score and not learner_death and matches
        reason = f"successful_{expected_route}" if success else "rejected_route_window"
        for window_index in range(start, index + 1):
            selected[window_index] = success
            reasons[window_index] = reason
        if success:
            route_counts[expected_route] += 1
        start = index + 1
    return ExplorerWindowLabels(
        selected=tuple(selected),
        reasons=tuple(reasons),
        successful_windows=sum(route_counts.values()),
        route_windows=tuple((route, route_counts[route]) for route in EXPLORER_ROUTE_CYCLE),
    )


def _matches_route(route: str, visited: set[str]) -> bool:
    if route == "flank":
        return _FLANK_REGIONS.issubset(visited)
    if not visited.isdisjoint(_FLANK_REGIONS):
        return False
    if route == "direct_upper":
        return "upper_route" in visited
    return "lower_route" in visited and "upper_route" not in visited


def _has_event(
    events: tuple[DuelEvent, ...], side: str, event_type: DuelEventType
) -> bool:
    return any(event.side == side and event.type is event_type for event in events)
