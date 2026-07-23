from __future__ import annotations

import math
from enum import Enum
from typing import Protocol

import numpy as np

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.defensive_signals import (
    defensive_risk,
    in_defensive_half,
    opponent_carrier_id,
)
from botcolosseo.envs.duel_types import DuelPrivilegedState
from botcolosseo.scenarios.regions import RegionGraph


class DuelTeacherMode(str, Enum):
    OBJECTIVE = "objective"
    INTERCEPT = "intercept"
    EVADE = "evade"
    RECOVER = "recover"
    DEFEND = "defend"
    RANDOM = "random"


class DuelTeacher(Protocol):
    name: str
    side: str
    mode: DuelTeacherMode

    def reset(self, *, seed: int) -> None: ...

    def act(self, state: DuelPrivilegedState) -> MacroAction: ...


def _player(state: DuelPrivilegedState, side: str) -> tuple[float, float, float]:
    if side == "host":
        return state.host_x, state.host_y, state.host_angle
    return state.opponent_x, state.opponent_y, state.opponent_angle


def _opponent(state: DuelPrivilegedState, side: str) -> tuple[float, float]:
    if side == "host":
        return state.opponent_x, state.opponent_y
    return state.host_x, state.host_y


def _health(state: DuelPrivilegedState, side: str) -> tuple[float, float]:
    if side == "host":
        return state.host_health, state.opponent_health
    return state.opponent_health, state.host_health


def _base(side: str) -> tuple[float, float]:
    return (-640.0, 0.0) if side == "host" else (640.0, 0.0)


def _recover(state: DuelPrivilegedState, side: str) -> MacroAction:
    x, y, _ = _player(state, side)
    target = _base(side)
    if math.hypot(target[0] - x, target[1] - y) <= 32.0:
        return MacroAction.IDLE
    return _steer(state, side, target)


def _steer(
    state: DuelPrivilegedState, side: str, target: tuple[float, float]
) -> MacroAction:
    x, y, angle = _player(state, side)
    desired = math.degrees(math.atan2(target[1] - y, target[0] - x)) % 360.0
    error = (desired - angle + 180.0) % 360.0 - 180.0
    if error > 45.0:
        return MacroAction.TURN_LEFT
    if error < -45.0:
        return MacroAction.TURN_RIGHT
    if error > 12.0:
        return MacroAction.FORWARD_TURN_LEFT
    if error < -12.0:
        return MacroAction.FORWARD_TURN_RIGHT
    return MacroAction.MOVE_FORWARD


class ObjectiveDuelTeacher:
    name = "objective_first"

    def __init__(self, graph: RegionGraph, *, side: str) -> None:
        del graph
        if side not in ("host", "opponent"):
            raise ValueError(f"Invalid duel side: {side}")
        self.side = side
        self.mode = DuelTeacherMode.OBJECTIVE

    def reset(self, *, seed: int) -> None:
        del seed
        self.mode = DuelTeacherMode.OBJECTIVE

    def act(self, state: DuelPrivilegedState) -> MacroAction:
        own_health, _ = _health(state, self.side)
        if own_health <= 25.0:
            self.mode = DuelTeacherMode.RECOVER
            return _recover(state, self.side)
        own_carrier = 1 if self.side == "host" else 2
        if state.carrier == own_carrier:
            self.mode = DuelTeacherMode.EVADE
            target = _base(self.side)
        else:
            self.mode = DuelTeacherMode.OBJECTIVE
            target = (state.core_x, state.core_y)
        return _steer(state, self.side, target)


class FixedRouteDuelTeacher:
    name = "fixed_route"

    def __init__(self, graph: RegionGraph, *, side: str) -> None:
        if side not in ("host", "opponent"):
            raise ValueError(f"Invalid duel side: {side}")
        self.side = side
        points = graph.route("direct_upper").waypoints
        self._points = points if side == "host" else tuple((-x, y) for x, y in points)
        self._index = 0
        self.mode = DuelTeacherMode.OBJECTIVE

    def reset(self, *, seed: int) -> None:
        del seed
        self._index = 0
        self.mode = DuelTeacherMode.OBJECTIVE

    def act(self, state: DuelPrivilegedState) -> MacroAction:
        self.mode = DuelTeacherMode.OBJECTIVE
        x, y, _ = _player(state, self.side)
        while self._index < len(self._points) - 1:
            target = self._points[self._index]
            if math.hypot(target[0] - x, target[1] - y) > 48.0:
                break
            self._index += 1
        return _steer(state, self.side, self._points[self._index])


class AggressiveDuelTeacher:
    name = "aggressive_script"

    def __init__(self, graph: RegionGraph, *, side: str) -> None:
        del graph
        if side not in ("host", "opponent"):
            raise ValueError(f"Invalid duel side: {side}")
        self.side = side
        self.mode = DuelTeacherMode.INTERCEPT

    def reset(self, *, seed: int) -> None:
        del seed
        self.mode = DuelTeacherMode.INTERCEPT

    def act(self, state: DuelPrivilegedState) -> MacroAction:
        own_health, opponent_health = _health(state, self.side)
        if own_health <= 0.0:
            self.mode = DuelTeacherMode.RECOVER
            return MacroAction.IDLE
        if own_health <= 25.0 and opponent_health > own_health:
            self.mode = DuelTeacherMode.EVADE
            return _recover(state, self.side)
        self.mode = DuelTeacherMode.INTERCEPT
        x, y, angle = _player(state, self.side)
        target = _opponent(state, self.side)
        distance = math.hypot(target[0] - x, target[1] - y)
        desired = math.degrees(math.atan2(target[1] - y, target[0] - x)) % 360.0
        error = (desired - angle + 180.0) % 360.0 - 180.0
        if distance <= 512.0:
            if error > 8.0:
                return MacroAction.TURN_LEFT_ATTACK
            if error < -8.0:
                return MacroAction.TURN_RIGHT_ATTACK
            if distance > 96.0:
                return MacroAction.FORWARD_ATTACK
            return MacroAction.ATTACK
        return _steer(state, self.side, target)


class DefensiveDuelTeacher:
    name = "defensive_script"

    def __init__(self, graph: RegionGraph, *, side: str) -> None:
        if side not in ("host", "opponent"):
            raise ValueError(f"Invalid duel side: {side}")
        self.side = side
        self._base = _base(side)
        self._aggressive = AggressiveDuelTeacher(graph, side=side)
        self.mode = DuelTeacherMode.DEFEND

    def reset(self, *, seed: int) -> None:
        self._aggressive.reset(seed=seed)
        self.mode = DuelTeacherMode.DEFEND

    def act(self, state: DuelPrivilegedState) -> MacroAction:
        own_health, _ = _health(state, self.side)
        if own_health <= 0.0:
            self.mode = DuelTeacherMode.RECOVER
            return MacroAction.IDLE
        opponent_carrier = 2 if self.side == "host" else 1
        if state.carrier == opponent_carrier:
            self.mode = DuelTeacherMode.INTERCEPT
            return self._aggressive.act(state)
        self.mode = DuelTeacherMode.DEFEND
        x, y, _ = _player(state, self.side)
        if math.hypot(self._base[0] - x, self._base[1] - y) <= 32.0:
            return MacroAction.IDLE
        return _steer(state, self.side, self._base)


EXPLORER_ROUTE_CYCLE = ("direct_upper", "direct_lower", "flank")


class RouteExplorerTeacher:
    """Training-only Teacher that cycles routes using the public own score."""

    name = "route_explorer_teacher"

    def __init__(self, graph: RegionGraph, *, side: str) -> None:
        if side not in ("host", "opponent"):
            raise ValueError(f"Invalid duel side: {side}")
        self.side = side
        self._graph = graph
        self._initial_score: int | None = None
        self._key: tuple[int, bool] | None = None
        self._points: tuple[tuple[float, float], ...] = ()
        self._index = 0
        self.route_name = EXPLORER_ROUTE_CYCLE[0]
        self.mode = DuelTeacherMode.OBJECTIVE

    def reset(self, *, seed: int) -> None:
        del seed
        self._initial_score = None
        self._key = None
        self._points = ()
        self._index = 0
        self.route_name = EXPLORER_ROUTE_CYCLE[0]
        self.mode = DuelTeacherMode.OBJECTIVE

    def act(self, state: DuelPrivilegedState) -> MacroAction:
        score = state.host_score if self.side == "host" else state.opponent_score
        if self._initial_score is None:
            self._initial_score = score
        if score < self._initial_score:
            raise ValueError("Explorer own score decreased within an episode")
        score_progress = score - self._initial_score
        return self.act_for_mode(
            state, score_progress % len(EXPLORER_ROUTE_CYCLE)
        )

    def act_for_mode(
        self, state: DuelPrivilegedState, route_mode: int
    ) -> MacroAction:
        if not 0 <= route_mode < len(EXPLORER_ROUTE_CYCLE):
            raise ValueError("Invalid Explorer route mode")
        carrier = 1 if self.side == "host" else 2
        carrying = state.carrier == carrier
        key = (route_mode, carrying)
        if key != self._key:
            self.route_name = EXPLORER_ROUTE_CYCLE[route_mode]
            points = self._graph.route(self.route_name).waypoints
            if self.side == "opponent":
                points = tuple((-x, y) for x, y in points)
            self._points = (
                (*reversed(points), _base(self.side)) if carrying else points
            )
            self._index = 0
            self._key = key
        x, y, _ = _player(state, self.side)
        while self._index < len(self._points) - 1:
            target = self._points[self._index]
            if math.hypot(target[0] - x, target[1] - y) > 48.0:
                break
            self._index += 1
        self.mode = DuelTeacherMode.EVADE if carrying else DuelTeacherMode.OBJECTIVE
        return _steer(state, self.side, self._points[self._index])


class ProtectiveDefensiveTeacher:
    """Training-only Teacher that defends only when the state warrants it."""

    name = "protective_defensive_teacher"

    def __init__(self, graph: RegionGraph, *, side: str) -> None:
        if side not in ("host", "opponent"):
            raise ValueError(f"Invalid duel side: {side}")
        self.side = side
        self._objective = ObjectiveDuelTeacher(graph, side=side)
        self._aggressive = AggressiveDuelTeacher(graph, side=side)
        self.mode = DuelTeacherMode.OBJECTIVE

    def reset(self, *, seed: int) -> None:
        self._objective.reset(seed=seed)
        self._aggressive.reset(seed=seed)
        self.mode = DuelTeacherMode.OBJECTIVE

    def act(self, state: DuelPrivilegedState) -> MacroAction:
        own_health, opponent_health = _health(state, self.side)
        if own_health <= 0.0:
            self.mode = DuelTeacherMode.RECOVER
            return MacroAction.IDLE
        if own_health <= 25.0 and opponent_health > own_health:
            self.mode = DuelTeacherMode.EVADE
            return _recover(state, self.side)
        if state.carrier == opponent_carrier_id(self.side):
            self.mode = DuelTeacherMode.INTERCEPT
            return self._aggressive.act(state)
        if state.carrier == 0 and in_defensive_half(state.core_x, self.side):
            self.mode = DuelTeacherMode.DEFEND
            return _steer(state, self.side, (state.core_x, state.core_y))
        if defensive_risk(state, self.side):
            self.mode = DuelTeacherMode.INTERCEPT
            return self._aggressive.act(state)
        self.mode = DuelTeacherMode.OBJECTIVE
        return self._objective.act(state)


class RandomDuelTeacher:
    name = "random_legal"

    def __init__(self, graph: RegionGraph, *, side: str) -> None:
        del graph
        if side not in ("host", "opponent"):
            raise ValueError(f"Invalid duel side: {side}")
        self.side = side
        self._rng: np.random.Generator | None = None
        self.mode = DuelTeacherMode.RANDOM

    def reset(self, *, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self.mode = DuelTeacherMode.RANDOM

    def act(self, state: DuelPrivilegedState) -> MacroAction:
        del state
        self.mode = DuelTeacherMode.RANDOM
        if self._rng is None:
            raise RuntimeError("RandomDuelTeacher must be reset before act")
        return MacroAction(int(self._rng.integers(0, len(MacroAction))))


DUEL_TEACHERS = {
    teacher.name: teacher
    for teacher in (
        RandomDuelTeacher,
        FixedRouteDuelTeacher,
        ObjectiveDuelTeacher,
        AggressiveDuelTeacher,
        DefensiveDuelTeacher,
    )
}


def create_duel_teacher(name: str, graph: RegionGraph, *, side: str) -> DuelTeacher:
    try:
        teacher_type = DUEL_TEACHERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown duel Teacher: {name}") from exc
    return teacher_type(graph, side=side)
