from __future__ import annotations

import math

from botcolosseo.envs.duel_types import DuelPrivilegedState

DEFENSIVE_HALF_BOUNDARY = 128.0
PROTECTIVE_ZONE_RADIUS = 512.0


def learner_carrier_id(side: str) -> int:
    _validate_side(side)
    return 1 if side == "host" else 2


def opponent_carrier_id(side: str) -> int:
    _validate_side(side)
    return 2 if side == "host" else 1


def home_point(side: str) -> tuple[float, float]:
    _validate_side(side)
    return (-640.0, 0.0) if side == "host" else (640.0, 0.0)


def learner_position(state: DuelPrivilegedState, side: str) -> tuple[float, float]:
    _validate_side(side)
    return (
        (state.host_x, state.host_y)
        if side == "host"
        else (state.opponent_x, state.opponent_y)
    )


def opposing_position(state: DuelPrivilegedState, side: str) -> tuple[float, float]:
    _validate_side(side)
    return (
        (state.opponent_x, state.opponent_y)
        if side == "host"
        else (state.host_x, state.host_y)
    )


def in_defensive_half(x: float, side: str) -> bool:
    _validate_side(side)
    return x < -DEFENSIVE_HALF_BOUNDARY if side == "host" else x > DEFENSIVE_HALF_BOUNDARY


def defensive_risk(state: DuelPrivilegedState, side: str) -> bool:
    if state.carrier == opponent_carrier_id(side):
        return True
    if state.carrier == 0 and in_defensive_half(state.core_x, side):
        return True
    opponent_x, _ = opposing_position(state, side)
    return state.carrier != learner_carrier_id(side) and in_defensive_half(
        opponent_x, side
    )


def in_protective_zone(state: DuelPrivilegedState, side: str) -> bool:
    x, y = learner_position(state, side)
    home_x, home_y = home_point(side)
    return math.hypot(x - home_x, y - home_y) <= PROTECTIVE_ZONE_RADIUS


def unnecessary_guard(state: DuelPrivilegedState, side: str) -> bool:
    return (
        state.carrier != learner_carrier_id(side)
        and in_protective_zone(state, side)
        and not defensive_risk(state, side)
    )


def _validate_side(side: str) -> None:
    if side not in ("host", "opponent"):
        raise ValueError(f"Invalid duel side: {side}")
