from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.types import PrivilegedState
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.scenarios.splits import TaskKind


class Teacher(Protocol):
    name: str

    def reset(self, *, seed: int, task: TaskKind) -> None: ...

    def act(self, state: PrivilegedState) -> MacroAction: ...


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def _angle_error(
    player_x: float,
    player_y: float,
    player_angle: float,
    target_x: float,
    target_y: float,
) -> float:
    desired = math.degrees(math.atan2(target_y - player_y, target_x - player_x)) % 360.0
    return (desired - player_angle + 180.0) % 360.0 - 180.0


def _steer_toward(state: PrivilegedState, target: tuple[float, float]) -> MacroAction:
    error = _angle_error(
        state.player_x,
        state.player_y,
        state.player_angle,
        target[0],
        target[1],
    )
    if error > 45.0:
        return MacroAction.TURN_LEFT
    if error < -45.0:
        return MacroAction.TURN_RIGHT
    if error > 12.0:
        return MacroAction.FORWARD_TURN_LEFT
    if error < -12.0:
        return MacroAction.FORWARD_TURN_RIGHT
    return MacroAction.MOVE_FORWARD


class _WaypointFollower:
    def __init__(
        self,
        waypoints: tuple[tuple[float, float], ...],
        *,
        arrival_tolerance: float = 48.0,
    ) -> None:
        if not waypoints:
            raise ValueError("Waypoint follower requires at least one waypoint")
        self._waypoints = waypoints
        self._arrival_tolerance = arrival_tolerance
        self._index = 0

    def reset(self, *, nearest_to: PrivilegedState | None = None) -> None:
        if nearest_to is None:
            self._index = 0
            return
        self._index = min(
            range(len(self._waypoints)),
            key=lambda index: _distance(
                nearest_to.player_x,
                nearest_to.player_y,
                *self._waypoints[index],
            ),
        )

    def act(self, state: PrivilegedState) -> MacroAction:
        while self._index < len(self._waypoints) - 1:
            waypoint = self._waypoints[self._index]
            if (
                _distance(state.player_x, state.player_y, *waypoint)
                > self._arrival_tolerance
            ):
                break
            self._index += 1
        target = self._waypoints[self._index]
        if (
            self._index == len(self._waypoints) - 1
            and _distance(state.player_x, state.player_y, *target)
            <= self._arrival_tolerance
        ):
            return MacroAction.IDLE
        return _steer_toward(state, target)


class FixedRouteTeacher:
    name = "fixed_route"

    def __init__(self, graph: RegionGraph, *, route_name: str = "direct_lower") -> None:
        self._follower = _WaypointFollower(graph.route(route_name).waypoints)

    def reset(self, *, seed: int, task: TaskKind) -> None:
        self._follower.reset()

    def act(self, state: PrivilegedState) -> MacroAction:
        return self._follower.act(state)


class ObjectiveFirstTeacher:
    name = "objective_first"

    def __init__(self, graph: RegionGraph) -> None:
        return_waypoints = tuple(reversed(graph.route("direct_lower").waypoints))
        self._return_follower = _WaypointFollower(return_waypoints)
        self.phase = "SEARCH_CORE"

    def reset(self, *, seed: int, task: TaskKind) -> None:
        self.phase = "SEARCH_CORE"
        self._return_follower.reset()

    def act(self, state: PrivilegedState) -> MacroAction:
        if state.has_core:
            if self.phase != "RETURN_BASE":
                self.phase = "RETURN_BASE"
                self._return_follower.reset(nearest_to=state)
            return self._return_follower.act(state)
        self.phase = "SEARCH_CORE"
        return _steer_toward(state, (state.core_x, state.core_y))


class AggressiveScriptTeacher:
    name = "aggressive_script"

    def reset(self, *, seed: int, task: TaskKind) -> None:
        return None

    def act(self, state: PrivilegedState) -> MacroAction:
        if not state.target_alive:
            return MacroAction.IDLE
        distance = _distance(
            state.player_x,
            state.player_y,
            state.target_x,
            state.target_y,
        )
        error = _angle_error(
            state.player_x,
            state.player_y,
            state.player_angle,
            state.target_x,
            state.target_y,
        )
        if distance <= 512.0 and abs(error) <= 8.0:
            return MacroAction.ATTACK
        return _steer_toward(state, (state.target_x, state.target_y))


class DefensiveScriptTeacher:
    name = "defensive_script"

    def __init__(self, graph: RegionGraph) -> None:
        self._hold_point = (-600.0, 0.0)
        self._engage = AggressiveScriptTeacher()

    def reset(self, *, seed: int, task: TaskKind) -> None:
        self._engage.reset(seed=seed, task=task)

    def act(self, state: PrivilegedState) -> MacroAction:
        target_distance = _distance(
            state.player_x,
            state.player_y,
            state.target_x,
            state.target_y,
        )
        if state.target_alive and target_distance <= 512.0:
            return self._engage.act(state)
        if _distance(state.player_x, state.player_y, *self._hold_point) <= 32.0:
            return MacroAction.IDLE
        return _steer_toward(state, self._hold_point)


class EvasiveReturnTeacher:
    name = "evasive_return"

    def __init__(self, graph: RegionGraph) -> None:
        reversed_flank = tuple(reversed(graph.route("flank").waypoints))
        home = (-640.0, 0.0)
        self._follower = _WaypointFollower((*reversed_flank, home))
        self._decision = 0
        self._initialized = False

    def reset(self, *, seed: int, task: TaskKind) -> None:
        self._decision = 0
        self._initialized = False
        self._follower.reset()

    def act(self, state: PrivilegedState) -> MacroAction:
        if not self._initialized:
            self._follower.reset(nearest_to=state)
            self._initialized = True
        self._decision += 1
        if self._decision % 8 == 0:
            return (
                MacroAction.TURN_LEFT
                if (self._decision // 8) % 2
                else MacroAction.TURN_RIGHT
            )
        return self._follower.act(state)


class RandomLegal:
    name = "random_legal"

    def __init__(self) -> None:
        self._rng: np.random.Generator | None = None

    def reset(self, *, seed: int, task: TaskKind) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, state: PrivilegedState) -> MacroAction:
        if self._rng is None:
            raise RuntimeError("RandomLegal must be reset before act")
        return MacroAction(int(self._rng.integers(0, len(MacroAction))))


TEACHER_REGISTRY = {
    FixedRouteTeacher.name: FixedRouteTeacher,
    ObjectiveFirstTeacher.name: ObjectiveFirstTeacher,
    AggressiveScriptTeacher.name: AggressiveScriptTeacher,
    DefensiveScriptTeacher.name: DefensiveScriptTeacher,
    EvasiveReturnTeacher.name: EvasiveReturnTeacher,
}


def create_teacher(name: str, graph: RegionGraph) -> Teacher:
    if name not in TEACHER_REGISTRY:
        raise ValueError(f"Unknown Teacher: {name}")
    teacher_type = TEACHER_REGISTRY[name]
    if teacher_type is AggressiveScriptTeacher:
        return teacher_type()
    return teacher_type(graph)
