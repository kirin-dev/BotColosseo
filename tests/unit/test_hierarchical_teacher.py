from dataclasses import replace
from types import SimpleNamespace

import pytest

from botcolosseo.agents.hierarchical_teacher import (
    PrivilegedCommandTeacher,
    counterfactual_command_labels,
)
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_types import ExtractionPrivilegedState
from botcolosseo.training.hierarchical_protocol import Command


@pytest.fixture
def state():
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
        round_state=0,
        winner=0,
        engine_tic=0,
    )


def test_search_skips_missing_loot_and_requires_pickup(state):
    teacher = PrivilegedCommandTeacher(side="host", layout_variant=0)
    teacher.begin(Command.SEARCH_NORTH, state)
    assert teacher.act(state).status == "executing"
    assert teacher.act(replace(state, world_loot_mask=0)).status == "unavailable"
    # Carried value alone cannot attribute a pickup to the commanded region.
    assert teacher.act(replace(state, host_slots=(10, 0, 0))).status != "complete"


def test_extraction_arrival_is_not_success(state):
    teacher = PrivilegedCommandTeacher(side="host", layout_variant=0)
    teacher.begin(Command.EXTRACT_NORTH, state)
    decision = teacher.act(replace(state, host_x=0, host_y=400))
    assert decision.action == MacroAction.IDLE
    assert decision.status == "executing"
    extracted = replace(state, host_banked=25, host_health=0, host_x=0, host_y=400)
    assert teacher.act(extracted).status == "complete"
    assert teacher.act(replace(extracted, host_y=-400)).status == "inactive"


def test_engage_completion_requires_alive_target_at_start(state):
    teacher = PrivilegedCommandTeacher(side="host", layout_variant=0)
    teacher.begin(Command.ENGAGE, state)
    dead = replace(state, opponent_health=0)
    assert teacher.act(dead).status == "complete"
    teacher.begin(Command.ENGAGE, dead)
    assert teacher.act(dead).status == "unavailable"


def test_disengage_translates_instead_of_turning_under_fire(state):
    contact = replace(state, host_x=-200, opponent_x=200, host_angle=0)
    teacher = PrivilegedCommandTeacher(side="host", layout_variant=0)
    teacher.begin(Command.DISENGAGE, contact)
    assert teacher.act(contact).action == MacroAction.MOVE_BACKWARD
    sideways = replace(contact, host_angle=90)
    assert teacher.act(sideways).action in (MacroAction.STRAFE_LEFT, MacroAction.STRAFE_RIGHT)


def test_reset_and_switch_clear_command_baseline(state):
    teacher = PrivilegedCommandTeacher(side="host", layout_variant=0)
    teacher.begin(Command.SEARCH_NORTH, state)
    loaded = replace(state, host_slots=(10, 0, 0))
    teacher.begin(Command.SEARCH_SOUTH, loaded)
    assert teacher.act(loaded).status != "complete"
    teacher.reset()
    with pytest.raises(ValueError):
        teacher.act(state)


@pytest.mark.parametrize(
    "loot_id,side,total,expected",
    [
        (3, 1, 1, True),
        (2, 1, 1, False),
        (3, 2, 1, False),
        (3, 1, 2, False),
        (0, 1, 1, False),
    ],
)
def test_search_pickup_attribution(state, loot_id, side, total, expected):
    teacher = PrivilegedCommandTeacher(side="host", layout_variant=0)
    teacher.begin(Command.SEARCH_NORTH, state)
    before = SimpleNamespace(host_loot_pickups=0, opponent_loot_pickups=0, world_loot_mask=127)
    after = SimpleNamespace(
        host_loot_pickups=1,
        opponent_loot_pickups=total - 1,
        world_loot_mask=127 & ~(1 << max(0, loot_id - 1)),
        last_loot_id=loot_id,
        last_loot_side=side,
    )
    teacher.observe_transition(before, after)
    assert teacher.search_completed is expected


def test_counterfactual_labels_separate_opposite_routes_without_fake_targets(state):
    labels, valid = counterfactual_command_labels(state, side="host", layout_variant=0)
    assert len(labels) == len(valid) == 7
    assert valid[Command.EXTRACT_NORTH] and valid[Command.EXTRACT_SOUTH]
    assert labels[Command.EXTRACT_NORTH] != labels[Command.EXTRACT_SOUTH]
    assert not valid[Command.DISENGAGE]  # Initially far apart: no forced evasion.
