from pathlib import Path

from botcolosseo.agents.duel_teachers import (
    AggressiveDuelTeacher,
    DefensiveDuelTeacher,
    DuelTeacherMode,
    FixedRouteDuelTeacher,
    ObjectiveDuelTeacher,
    RandomDuelTeacher,
)
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_types import DuelPrivilegedState
from botcolosseo.scenarios.regions import RegionGraph


def state(**changes) -> DuelPrivilegedState:
    payload = {
        "host_x": -640.0,
        "host_y": 0.0,
        "host_angle": 0.0,
        "host_region": "home",
        "opponent_x": 640.0,
        "opponent_y": 0.0,
        "opponent_angle": 180.0,
        "opponent_region": "away",
        "core_x": 0.0,
        "core_y": 0.0,
        "carrier": 0,
        "host_health": 100.0,
        "opponent_health": 100.0,
        "host_score": 0,
        "opponent_score": 0,
        "round_state": 1,
        "engine_tic": 10,
    }
    payload.update(changes)
    return DuelPrivilegedState(**payload)


def graph() -> RegionGraph:
    return RegionGraph.from_yaml(Path("assets/scenarios/crystal_run/src/regions.yaml"))


def test_objective_teacher_seeks_core_then_own_base() -> None:
    teacher = ObjectiveDuelTeacher(graph(), side="host")
    teacher.reset(seed=7)

    assert teacher.act(state()) == MacroAction.MOVE_FORWARD
    assert teacher.mode is DuelTeacherMode.OBJECTIVE
    carrying = state(host_x=0.0, carrier=1)
    assert teacher.act(carrying) in {
        MacroAction.TURN_LEFT,
        MacroAction.TURN_RIGHT,
        MacroAction.FORWARD_TURN_LEFT,
        MacroAction.FORWARD_TURN_RIGHT,
    }
    assert teacher.mode is DuelTeacherMode.EVADE
    teacher.act(state(host_health=20.0))
    assert teacher.mode is DuelTeacherMode.RECOVER


def test_aggressive_attacks_aligned_opponent_and_defensive_holds_base() -> None:
    aggressive = AggressiveDuelTeacher(graph(), side="host")
    defensive = DefensiveDuelTeacher(graph(), side="host")
    aggressive.reset(seed=7)
    defensive.reset(seed=7)
    nearby = state(opponent_x=-256.0)

    assert aggressive.act(nearby) == MacroAction.FORWARD_ATTACK
    assert aggressive.mode is DuelTeacherMode.INTERCEPT
    angled = state(opponent_x=-256.0, opponent_y=256.0)
    assert aggressive.act(angled) == MacroAction.TURN_LEFT_ATTACK
    assert defensive.act(state()) == MacroAction.IDLE
    assert defensive.mode is DuelTeacherMode.DEFEND
    assert defensive.act(state(carrier=2)) != MacroAction.IDLE
    assert defensive.mode is DuelTeacherMode.INTERCEPT
    aggressive.act(state(host_health=10.0))
    assert aggressive.mode is DuelTeacherMode.EVADE
    aggressive.act(state(host_health=0.0))
    assert aggressive.mode is DuelTeacherMode.RECOVER


def test_fixed_route_and_random_are_deterministic() -> None:
    fixed = FixedRouteDuelTeacher(graph(), side="host")
    first = RandomDuelTeacher(graph(), side="host")
    second = RandomDuelTeacher(graph(), side="host")
    fixed.reset(seed=9)
    first.reset(seed=9)
    second.reset(seed=9)

    assert fixed.act(state()) in set(MacroAction)
    assert [first.act(state()) for _ in range(20)] == [second.act(state()) for _ in range(20)]
    assert first.mode is DuelTeacherMode.RANDOM
