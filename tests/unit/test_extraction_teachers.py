from __future__ import annotations

from dataclasses import replace

from botcolosseo.agents.extraction_teachers import (
    AggressiveExtractionTeacher,
    ExtractionStyle,
    ExtractionWaypointTeacher,
    StyledExtractionTeacher,
)
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_types import ExtractionPrivilegedState


def state() -> ExtractionPrivilegedState:
    return ExtractionPrivilegedState(
        host_x=-640,
        host_y=0,
        host_angle=0,
        opponent_x=640,
        opponent_y=0,
        opponent_angle=180,
        host_health=100,
        opponent_health=100,
        host_slots=(0, 0, 0),
        opponent_slots=(0, 0, 0),
        host_banked=0,
        opponent_banked=0,
        cache_owner=0,
        cache_slots=(0, 0, 0),
        cache_x=0,
        cache_y=0,
        world_loot_mask=127,
        round_state=1,
        winner=0,
        engine_tic=0,
    )


def test_waypoint_teacher_advances_and_stops() -> None:
    teacher = ExtractionWaypointTeacher(
        side="host",
        waypoints=((-520, 0), (-224, 0)),
    )

    assert teacher.act(state()) is MacroAction.MOVE_FORWARD
    assert teacher.act(replace(state(), host_x=-520)) is MacroAction.MOVE_FORWARD
    assert teacher.act(replace(state(), host_x=-224)) is MacroAction.IDLE
    assert teacher.finished is True


def test_aggressive_teacher_closes_then_attacks() -> None:
    teacher = AggressiveExtractionTeacher(side="host")

    assert teacher.act(state()) is MacroAction.MOVE_FORWARD
    assert teacher.act(replace(state(), host_x=200)) is MacroAction.FORWARD_ATTACK


def test_styled_teachers_expose_distinct_initial_intentions() -> None:
    aggressive = StyledExtractionTeacher(
        side="host", style=ExtractionStyle.AGGRESSIVE
    )
    defensive = StyledExtractionTeacher(
        side="host", style=ExtractionStyle.DEFENSIVE
    )
    explorer = StyledExtractionTeacher(side="host", style=ExtractionStyle.EXPLORER)

    assert aggressive.style is ExtractionStyle.AGGRESSIVE
    assert defensive.style is ExtractionStyle.DEFENSIVE
    assert explorer.style is ExtractionStyle.EXPLORER
    assert aggressive.act(state()) is MacroAction.TURN_LEFT
    assert defensive.act(state()) is MacroAction.TURN_LEFT
    assert explorer.act(state()) is MacroAction.TURN_LEFT


def test_aggressive_profile_converts_death_to_cache_route() -> None:
    teacher = StyledExtractionTeacher(
        side="host", style=ExtractionStyle.AGGRESSIVE
    )
    carrying = replace(state(), host_slots=(10, 0, 0), host_x=200)

    assert teacher.act(carrying) is MacroAction.FORWARD_ATTACK

    cache = replace(
        carrying,
        opponent_health=0,
        cache_owner=2,
        cache_x=-128,
        cache_y=0,
    )
    assert teacher.act(cache) is MacroAction.TURN_RIGHT
