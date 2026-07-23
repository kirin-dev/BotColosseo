from pathlib import Path

from botcolosseo.agents.duel_teachers import (
    DuelTeacherMode,
    ProtectiveDefensiveTeacher,
)
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.defensive_signals import (
    defensive_risk,
    in_protective_zone,
    unnecessary_guard,
)
from botcolosseo.envs.duel_types import DuelPrivilegedState
from botcolosseo.scenarios.regions import RegionGraph


def _state(**overrides: object) -> DuelPrivilegedState:
    values: dict[str, object] = {
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
        "engine_tic": 0,
    }
    values.update(overrides)
    return DuelPrivilegedState(**values)  # type: ignore[arg-type]


def _graph() -> RegionGraph:
    return RegionGraph.from_yaml(Path("assets/scenarios/crystal_run/src/regions.yaml"))


def test_defensive_risk_is_side_mirrored() -> None:
    assert defensive_risk(_state(carrier=2), "host")
    assert defensive_risk(_state(carrier=1), "opponent")
    assert defensive_risk(_state(core_x=-200.0), "host")
    assert defensive_risk(_state(core_x=200.0), "opponent")
    assert defensive_risk(_state(opponent_x=-200.0), "host")
    assert defensive_risk(_state(host_x=200.0), "opponent")
    assert not defensive_risk(_state(), "host")
    assert not defensive_risk(_state(), "opponent")


def test_unnecessary_guard_requires_home_presence_without_risk() -> None:
    safe = _state()

    assert in_protective_zone(safe, "host")
    assert unnecessary_guard(safe, "host")
    assert not unnecessary_guard(_state(carrier=2), "host")
    assert not unnecessary_guard(_state(host_x=0.0), "host")
    assert not unnecessary_guard(_state(carrier=1), "host")


def test_protective_teacher_scores_when_safe_and_intercepts_carrier() -> None:
    teacher = ProtectiveDefensiveTeacher(_graph(), side="host")
    teacher.reset(seed=7)

    safe_action = teacher.act(_state())
    assert teacher.mode is DuelTeacherMode.OBJECTIVE
    assert safe_action is MacroAction.MOVE_FORWARD

    intercept_action = teacher.act(_state(carrier=2, opponent_x=-200.0))
    assert teacher.mode is DuelTeacherMode.INTERCEPT
    assert intercept_action in (
        MacroAction.ATTACK,
        MacroAction.FORWARD_ATTACK,
        MacroAction.TURN_LEFT_ATTACK,
        MacroAction.TURN_RIGHT_ATTACK,
    )


def test_protective_teacher_recovers_loose_core_and_disengages_low_health() -> None:
    teacher = ProtectiveDefensiveTeacher(_graph(), side="host")
    teacher.reset(seed=9)

    recover_action = teacher.act(
        _state(host_x=0.0, host_angle=180.0, core_x=-300.0)
    )
    assert teacher.mode is DuelTeacherMode.DEFEND
    assert recover_action is MacroAction.MOVE_FORWARD

    evade_action = teacher.act(
        _state(host_x=0.0, host_angle=180.0, host_health=20.0)
    )
    assert teacher.mode is DuelTeacherMode.EVADE
    assert evade_action is MacroAction.MOVE_FORWARD
