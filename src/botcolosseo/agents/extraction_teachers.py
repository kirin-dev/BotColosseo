from __future__ import annotations

import math

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_types import ExtractionPrivilegedState


def player_pose(
    state: ExtractionPrivilegedState,
    side: str,
) -> tuple[float, float, float]:
    if side == "host":
        return state.host_x, state.host_y, state.host_angle
    if side == "opponent":
        return state.opponent_x, state.opponent_y, state.opponent_angle
    raise ValueError(f"Unsupported extraction side: {side}")


def opponent_position(
    state: ExtractionPrivilegedState,
    side: str,
) -> tuple[float, float]:
    if side == "host":
        return state.opponent_x, state.opponent_y
    if side == "opponent":
        return state.host_x, state.host_y
    raise ValueError(f"Unsupported extraction side: {side}")


def steer_toward(
    state: ExtractionPrivilegedState,
    side: str,
    target: tuple[float, float],
    *,
    attack: bool = False,
) -> MacroAction:
    x, y, angle = player_pose(state, side)
    desired = math.degrees(math.atan2(target[1] - y, target[0] - x)) % 360.0
    error = (desired - angle + 180.0) % 360.0 - 180.0
    if error > 45.0:
        return (
            MacroAction.TURN_LEFT_ATTACK if attack else MacroAction.TURN_LEFT
        )
    if error < -45.0:
        return (
            MacroAction.TURN_RIGHT_ATTACK if attack else MacroAction.TURN_RIGHT
        )
    if error > 12.0:
        return (
            MacroAction.TURN_LEFT_ATTACK
            if attack
            else MacroAction.FORWARD_TURN_LEFT
        )
    if error < -12.0:
        return (
            MacroAction.TURN_RIGHT_ATTACK
            if attack
            else MacroAction.FORWARD_TURN_RIGHT
        )
    return MacroAction.FORWARD_ATTACK if attack else MacroAction.MOVE_FORWARD


class ExtractionWaypointTeacher:
    def __init__(
        self,
        *,
        side: str,
        waypoints: tuple[tuple[float, float], ...],
        arrival_tolerance: float = 36.0,
    ) -> None:
        if side not in ("host", "opponent"):
            raise ValueError(f"Unsupported extraction side: {side}")
        if not waypoints:
            raise ValueError("Extraction waypoint teacher requires a route")
        self.side = side
        self._waypoints = waypoints
        self._arrival_tolerance = arrival_tolerance
        self._index = 0

    @property
    def finished(self) -> bool:
        return self._index == len(self._waypoints)

    def reset(self) -> None:
        self._index = 0

    def act(self, state: ExtractionPrivilegedState) -> MacroAction:
        x, y, _ = player_pose(state, self.side)
        while self._index < len(self._waypoints):
            target = self._waypoints[self._index]
            if math.hypot(target[0] - x, target[1] - y) > self._arrival_tolerance:
                return steer_toward(state, self.side, target)
            self._index += 1
        return MacroAction.IDLE


class AggressiveExtractionTeacher:
    def __init__(self, *, side: str) -> None:
        if side not in ("host", "opponent"):
            raise ValueError(f"Unsupported extraction side: {side}")
        self.side = side

    def act(self, state: ExtractionPrivilegedState) -> MacroAction:
        target = opponent_position(state, self.side)
        x, y, _ = player_pose(state, self.side)
        distance = math.hypot(target[0] - x, target[1] - y)
        return steer_toward(
            state,
            self.side,
            target,
            attack=distance <= 512.0,
        )
