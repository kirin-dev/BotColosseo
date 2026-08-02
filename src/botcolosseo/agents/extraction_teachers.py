from __future__ import annotations

import math
from enum import Enum

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_layouts import (
    randomized_loot_layout,
    strong_randomized_route,
)
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


def player_slots(
    state: ExtractionPrivilegedState,
    side: str,
) -> tuple[int, int, int]:
    if side == "host":
        return state.host_slots
    if side == "opponent":
        return state.opponent_slots
    raise ValueError(f"Unsupported extraction side: {side}")


def player_health(state: ExtractionPrivilegedState, side: str) -> float:
    if side == "host":
        return state.host_health
    if side == "opponent":
        return state.opponent_health
    raise ValueError(f"Unsupported extraction side: {side}")


def opponent_health(state: ExtractionPrivilegedState, side: str) -> float:
    return player_health(state, "opponent" if side == "host" else "host")


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


class ExtractionStyle(str, Enum):
    STRONG = "strong"
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    EXPLORER = "explorer"


def _route(side: str, style: ExtractionStyle) -> tuple[tuple[float, float], ...]:
    sign = -1.0 if side == "host" else 1.0
    own_low_a = (sign * 520.0, 288.0)
    own_low_b = (sign * 520.0, -288.0)
    own_contested = (sign * 224.0, 0.0)
    away_contested = (-sign * 224.0, 0.0)
    away_low_a = (-sign * 520.0, 288.0)
    away_low_b = (-sign * 520.0, -288.0)
    safe_extract = (0.0, 400.0 if side == "host" else -400.0)
    alternate_extract = (0.0, -400.0 if side == "host" else 400.0)
    if style is ExtractionStyle.AGGRESSIVE:
        return (own_low_a,)
    if style is ExtractionStyle.DEFENSIVE:
        return (own_low_a, own_low_b, own_contested, safe_extract)
    if style is ExtractionStyle.EXPLORER:
        return (
            own_low_a,
            own_low_b,
            own_contested,
            (0.0, 0.0),
            away_contested,
            away_low_a,
            away_low_b,
            alternate_extract,
        )
    return (own_low_a, own_low_b, own_contested, (0.0, 0.0), safe_extract)


class StyledExtractionTeacher:
    """Privileged training oracle whose labels remain public-observation inputs."""

    def __init__(self, *, side: str, style: ExtractionStyle | str) -> None:
        if side not in ("host", "opponent"):
            raise ValueError(f"Unsupported extraction side: {side}")
        self.side = side
        self.style = ExtractionStyle(style)
        self._waypoints = ExtractionWaypointTeacher(
            side=side,
            waypoints=_route(side, self.style),
            arrival_tolerance=28.0,
        )

    def reset(self) -> None:
        self._waypoints.reset()

    def act(self, state: ExtractionPrivilegedState) -> MacroAction:
        if player_health(state, self.side) <= 0:
            return MacroAction.IDLE
        own_x, own_y, _ = player_pose(state, self.side)
        target = opponent_position(state, self.side)
        distance = math.hypot(target[0] - own_x, target[1] - own_y)
        enemy_alive = opponent_health(state, self.side) > 0

        if self.style is ExtractionStyle.AGGRESSIVE:
            if sum(player_slots(state, self.side)) == 0 and not self._waypoints.finished:
                return self._waypoints.act(state)
            if enemy_alive:
                return steer_toward(
                    state,
                    self.side,
                    target,
                    attack=distance <= 512.0,
                )
            if state.cache_owner:
                return steer_toward(
                    state,
                    self.side,
                    (state.cache_x, state.cache_y),
                )
            extraction = (0.0, 400.0 if self.side == "host" else -400.0)
            return steer_toward(state, self.side, extraction)

        if self.style is ExtractionStyle.DEFENSIVE and enemy_alive and distance < 256:
            extraction = (0.0, 400.0 if self.side == "host" else -400.0)
            return steer_toward(state, self.side, extraction)

        if (
            self.style is ExtractionStyle.STRONG
            and enemy_alive
            and distance <= 384.0
            and player_health(state, self.side) >= 60
        ):
            return steer_toward(
                state,
                self.side,
                target,
                attack=True,
            )

        return self._waypoints.act(state)


class PrivilegedStrongExtractionTeacher:
    """Training-only expert kept separate from the scripted opponent pool."""

    def __init__(
        self,
        *,
        side: str,
        combat_budget: int = 48,
        layout_variant: int | None = None,
    ) -> None:
        if side not in ("host", "opponent"):
            raise ValueError(f"Unsupported extraction side: {side}")
        if combat_budget <= 0:
            raise ValueError("Strong Teacher combat budget must be positive")
        self.side = side
        self.style = ExtractionStyle.STRONG
        self._combat_budget = combat_budget
        self._combat_decisions = 0
        self._randomized_layout = (
            None if layout_variant is None else randomized_loot_layout(layout_variant)
        )
        self._loot_target_id: int | None = None
        route = (
            _route(side, ExtractionStyle.STRONG)
            if layout_variant is None
            else strong_randomized_route(side=side, variant=layout_variant)
        )
        self._waypoints = ExtractionWaypointTeacher(
            side=side,
            waypoints=route,
            arrival_tolerance=28.0,
        )

    def reset(self) -> None:
        self._combat_decisions = 0
        self._loot_target_id = None
        self._waypoints.reset()

    def _useful_loot_ids(
        self, state: ExtractionPrivilegedState
    ) -> tuple[int, ...]:
        if self._randomized_layout is None:
            return ()
        slots = player_slots(state, self.side)
        minimum = min(slots)
        return tuple(
            loot_id
            for loot_id, (value, _, _) in enumerate(self._randomized_layout)
            if state.world_loot_mask & (1 << loot_id) and value > minimum
        )

    def _randomized_loot_action(
        self, state: ExtractionPrivilegedState
    ) -> MacroAction | None:
        if self._randomized_layout is None:
            return None
        useful = self._useful_loot_ids(state)
        if self._loot_target_id not in useful:
            self._loot_target_id = None
        if self._loot_target_id is None and useful:
            own_x, own_y, _ = player_pose(state, self.side)
            minimum = min(player_slots(state, self.side))
            self._loot_target_id = min(
                useful,
                key=lambda loot_id: (
                    -(self._randomized_layout[loot_id][0] - minimum),
                    math.hypot(
                        self._randomized_layout[loot_id][1] - own_x,
                        self._randomized_layout[loot_id][2] - own_y,
                    ),
                    loot_id,
                ),
            )
        if self._loot_target_id is None:
            return None
        _, target_x, target_y = self._randomized_layout[self._loot_target_id]
        return steer_toward(state, self.side, (target_x, target_y))

    def act(self, state: ExtractionPrivilegedState) -> MacroAction:
        if player_health(state, self.side) <= 0:
            return MacroAction.IDLE
        own_x, own_y, _ = player_pose(state, self.side)
        carried = sum(player_slots(state, self.side))
        extraction = (0.0, 400.0 if self.side == "host" else -400.0)
        if carried >= 85 or (carried > 0 and state.engine_tic >= 1400):
            return steer_toward(state, self.side, extraction)

        enemy_alive = opponent_health(state, self.side) > 0
        if not enemy_alive and state.cache_owner and sum(state.cache_slots) > 0:
            return steer_toward(
                state,
                self.side,
                (state.cache_x, state.cache_y),
            )

        target = opponent_position(state, self.side)
        distance = math.hypot(target[0] - own_x, target[1] - own_y)
        if (
            enemy_alive
            and distance <= 384.0
            and self._combat_decisions < self._combat_budget
        ):
            self._combat_decisions += 1
            return steer_toward(state, self.side, target, attack=True)

        randomized_action = self._randomized_loot_action(state)
        if randomized_action is not None:
            return randomized_action

        if self._randomized_layout is not None or self._waypoints.finished:
            if math.hypot(extraction[0] - own_x, extraction[1] - own_y) <= 28.0:
                return MacroAction.IDLE
            return steer_toward(state, self.side, extraction)
        return self._waypoints.act(state)


def privileged_extraction_teacher(
    *,
    side: str,
    style: ExtractionStyle | str,
    layout_variant: int | None = None,
) -> StyledExtractionTeacher | PrivilegedStrongExtractionTeacher:
    selected = ExtractionStyle(style)
    if selected is ExtractionStyle.STRONG:
        return PrivilegedStrongExtractionTeacher(
            side=side,
            layout_variant=layout_variant,
        )
    return StyledExtractionTeacher(side=side, style=selected)
