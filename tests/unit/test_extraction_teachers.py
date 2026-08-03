from __future__ import annotations

from dataclasses import replace

from botcolosseo.agents.extraction_teachers import (
    AggressiveExtractionTeacher,
    ExtractionStyle,
    ExtractionWaypointTeacher,
    PrivilegedStrongExtractionTeacher,
    StyledExtractionTeacher,
)
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_layouts import randomized_loot_layout
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


def test_strong_profile_counterattacks_and_converts_cache() -> None:
    teacher = PrivilegedStrongExtractionTeacher(
        side="host",
    )
    encounter = replace(
        state(),
        host_x=256,
        host_health=40,
    )

    assert teacher.act(encounter) is MacroAction.FORWARD_ATTACK

    cache = replace(
        encounter,
        opponent_health=0,
        cache_owner=2,
        cache_slots=(50, 25, 10),
        cache_x=-128,
    )
    assert teacher.act(cache) is MacroAction.TURN_RIGHT


def test_strong_profile_extracts_high_value_backpack() -> None:
    teacher = PrivilegedStrongExtractionTeacher(
        side="host",
    )
    carrying = replace(
        state(),
        opponent_health=0,
        host_slots=(50, 25, 10),
    )

    assert teacher.act(carrying) is MacroAction.FORWARD_TURN_LEFT


def test_strong_profile_answers_close_threat_before_extracting() -> None:
    teacher = PrivilegedStrongExtractionTeacher(side="host")
    threatened = replace(
        state(),
        host_x=128,
        opponent_x=640,
        host_slots=(50, 25, 10),
    )

    assert teacher.act(threatened) is MacroAction.FORWARD_ATTACK


def test_privileged_strong_teacher_stops_unbounded_pursuit() -> None:
    teacher = PrivilegedStrongExtractionTeacher(side="host", combat_budget=2)
    encounter = replace(state(), host_x=256)

    assert teacher.act(encounter) is MacroAction.FORWARD_ATTACK
    assert teacher.act(encounter) is MacroAction.FORWARD_ATTACK
    assert teacher.act(encounter) is MacroAction.TURN_LEFT


def test_randomized_strong_teacher_invalidates_removed_target() -> None:
    teacher = PrivilegedStrongExtractionTeacher(side="host", layout_variant=0)
    layout = randomized_loot_layout(0)
    first = teacher.act(replace(state(), opponent_health=0))
    assert first in {
        MacroAction.MOVE_FORWARD,
        MacroAction.FORWARD_TURN_LEFT,
        MacroAction.FORWARD_TURN_RIGHT,
        MacroAction.TURN_LEFT,
        MacroAction.TURN_RIGHT,
    }
    target_id = teacher._loot_target_id
    assert target_id is not None

    removed = replace(
        state(),
        opponent_health=0,
        world_loot_mask=127 & ~(1 << target_id),
        host_x=layout[target_id][1],
        host_y=layout[target_id][2],
    )
    teacher.act(removed)
    assert teacher._loot_target_id != target_id


def test_randomized_strong_teacher_ignores_non_improving_loot() -> None:
    teacher = PrivilegedStrongExtractionTeacher(side="host", layout_variant=0)
    only_tens = sum(1 << loot_id for loot_id in range(4))
    full = replace(
        state(),
        opponent_health=0,
        host_slots=(25, 25, 50),
        world_loot_mask=only_tens,
    )

    assert teacher.act(full) is MacroAction.FORWARD_TURN_LEFT
    assert teacher._loot_target_id is None
