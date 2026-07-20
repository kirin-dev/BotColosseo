from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_types import DuelPrivilegedState
from botcolosseo.scenarios.regions import RegionGraph


class DuelTeacher(Protocol):
    name: str
    side: str

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

    def reset(self, *, seed: int) -> None:
        del seed

    def act(self, state: DuelPrivilegedState) -> MacroAction:
        own_carrier = 1 if self.side == "host" else 2
        if state.carrier == own_carrier:
            target = (-640.0, 0.0) if self.side == "host" else (640.0, 0.0)
        else:
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

    def reset(self, *, seed: int) -> None:
        del seed
        self._index = 0

    def act(self, state: DuelPrivilegedState) -> MacroAction:
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

    def reset(self, *, seed: int) -> None:
        del seed

    def act(self, state: DuelPrivilegedState) -> MacroAction:
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
        self._base = (-640.0, 0.0) if side == "host" else (640.0, 0.0)
        self._aggressive = AggressiveDuelTeacher(graph, side=side)

    def reset(self, *, seed: int) -> None:
        self._aggressive.reset(seed=seed)

    def act(self, state: DuelPrivilegedState) -> MacroAction:
        opponent_carrier = 2 if self.side == "host" else 1
        if state.carrier == opponent_carrier:
            return self._aggressive.act(state)
        x, y, _ = _player(state, self.side)
        if math.hypot(self._base[0] - x, self._base[1] - y) <= 32.0:
            return MacroAction.IDLE
        return _steer(state, self.side, self._base)


class RandomDuelTeacher:
    name = "random_legal"

    def __init__(self, graph: RegionGraph, *, side: str) -> None:
        del graph
        if side not in ("host", "opponent"):
            raise ValueError(f"Invalid duel side: {side}")
        self.side = side
        self._rng: np.random.Generator | None = None

    def reset(self, *, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, state: DuelPrivilegedState) -> MacroAction:
        del state
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
