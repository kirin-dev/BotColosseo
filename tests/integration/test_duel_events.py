from pathlib import Path

import pytest

from botcolosseo.agents.duel_teachers import (
    AggressiveDuelTeacher,
    ObjectiveDuelTeacher,
)
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_protocol import DuelEventType
from botcolosseo.envs.synchronous_duel import SynchronousDuelEnv
from botcolosseo.scenarios.regions import RegionGraph


def make_env(seed: int) -> tuple[SynchronousDuelEnv, RegionGraph]:
    graph = RegionGraph.from_yaml(Path("assets/scenarios/crystal_run/src/regions.yaml"))
    env = SynchronousDuelEnv(
        config_path=Path("assets/scenarios/crystal_run/crystal_run.cfg"),
        region_graph=graph,
        seed=seed,
    )
    return env, graph


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_objective_teacher_emits_real_pickup_and_score() -> None:
    env, graph = make_env(17)
    teacher = ObjectiveDuelTeacher(graph, side="host")
    seen: list[tuple[str, DuelEventType]] = []
    try:
        env.reset()
        teacher.reset(seed=17)
        for _ in range(400):
            step = env.step(teacher.act(env.teacher_state()), MacroAction.IDLE)
            seen.extend((event.side, event.type) for event in step.events)
            if ("host", DuelEventType.SCORE) in seen or step.truncated:
                break
    finally:
        env.close()

    assert ("host", DuelEventType.PICKUP) in seen
    assert ("host", DuelEventType.SCORE) in seen


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_aggressive_teachers_emit_valid_hit_but_idle_does_not() -> None:
    env, graph = make_env(19)
    host = AggressiveDuelTeacher(graph, side="host")
    opponent = AggressiveDuelTeacher(graph, side="opponent")
    seen: list[DuelEventType] = []
    try:
        env.reset()
        host.reset(seed=19)
        opponent.reset(seed=19)
        for _ in range(300):
            step = env.step(
                host.act(env.teacher_state()), opponent.act(env.teacher_state())
            )
            seen.extend(event.type for event in step.events)
            if DuelEventType.VALID_HIT in seen or step.truncated:
                break
    finally:
        env.close()
    assert DuelEventType.VALID_HIT in seen

    idle_env, _ = make_env(20)
    idle_seen: list[DuelEventType] = []
    try:
        idle_env.reset()
        for _ in range(30):
            step = idle_env.step(MacroAction.IDLE, MacroAction.IDLE)
            idle_seen.extend(event.type for event in step.events)
    finally:
        idle_env.close()
    assert DuelEventType.VALID_HIT not in idle_seen
