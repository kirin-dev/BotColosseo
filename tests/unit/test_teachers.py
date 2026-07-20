from pathlib import Path

from botcolosseo.agents.teachers import (
    AggressiveScriptTeacher,
    DefensiveScriptTeacher,
    EvasiveReturnTeacher,
    FixedRouteTeacher,
    ObjectiveFirstTeacher,
    RandomLegal,
)
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.types import PrivilegedState
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.scenarios.splits import TaskKind


def graph() -> RegionGraph:
    return RegionGraph.from_yaml(Path("assets/scenarios/crystal_run/src/regions.yaml"))


def state(**overrides) -> PrivilegedState:
    values = {
        "player_x": -640.0,
        "player_y": 0.0,
        "player_angle": 0.0,
        "region_name": "home",
        "core_x": 0.0,
        "core_y": 0.0,
        "target_x": 256.0,
        "target_y": 0.0,
        "target_alive": False,
        "task_phase": 2,
        "has_core": False,
    }
    values.update(overrides)
    return PrivilegedState(**values)


def test_fixed_route_turns_then_advances() -> None:
    teacher = FixedRouteTeacher(graph(), route_name="direct_upper")
    teacher.reset(seed=1, task=TaskKind.NAVIGATION)

    turn = teacher.act(state(player_angle=180.0))
    advance = teacher.act(state(player_angle=78.0))

    assert turn is MacroAction.TURN_RIGHT
    assert advance is MacroAction.MOVE_FORWARD


def test_objective_first_switches_to_return_when_carrying() -> None:
    teacher = ObjectiveFirstTeacher(graph())
    teacher.reset(seed=1, task=TaskKind.PICKUP)

    teacher.act(state(has_core=False))
    assert teacher.phase == "SEARCH_CORE"
    action = teacher.act(state(player_x=0.0, has_core=True))

    assert teacher.phase == "RETURN_BASE"
    assert action in {
        MacroAction.TURN_LEFT,
        MacroAction.TURN_RIGHT,
        MacroAction.FORWARD_TURN_LEFT,
        MacroAction.FORWARD_TURN_RIGHT,
        MacroAction.MOVE_FORWARD,
    }


def test_aggressive_attacks_only_with_valid_alignment_and_range() -> None:
    teacher = AggressiveScriptTeacher()
    teacher.reset(seed=1, task=TaskKind.STATIC_HIT)

    assert teacher.act(state(target_alive=False)) is MacroAction.IDLE
    assert teacher.act(state(target_alive=True, target_x=-128.0)) is MacroAction.ATTACK
    assert teacher.act(
        state(target_alive=True, target_x=-128.0, target_y=256.0)
    ) in {MacroAction.TURN_LEFT, MacroAction.FORWARD_TURN_LEFT}


def test_defensive_holds_without_target_and_delegates_engagement() -> None:
    teacher = DefensiveScriptTeacher(graph())
    teacher.reset(seed=1, task=TaskKind.NAVIGATION)

    assert teacher.act(state(player_x=-600.0, target_alive=False)) is MacroAction.IDLE
    assert teacher.act(
        state(player_x=-600.0, target_x=-128.0, target_alive=True)
    ) is MacroAction.ATTACK


def test_evasive_return_and_random_baseline_are_deterministic() -> None:
    first = EvasiveReturnTeacher(graph())
    second = EvasiveReturnTeacher(graph())
    first.reset(seed=9, task=TaskKind.RETURN)
    second.reset(seed=9, task=TaskKind.RETURN)
    states = [state(player_x=384.0, has_core=True, task_phase=4)] * 10

    assert [first.act(item) for item in states] == [second.act(item) for item in states]

    random_first = RandomLegal()
    random_second = RandomLegal()
    random_first.reset(seed=11, task=TaskKind.NAVIGATION)
    random_second.reset(seed=11, task=TaskKind.NAVIGATION)
    actions_first = [random_first.act(state()) for _ in range(20)]
    actions_second = [random_second.act(state()) for _ in range(20)]
    assert actions_first == actions_second
    assert all(0 <= action.value <= 12 for action in actions_first)
