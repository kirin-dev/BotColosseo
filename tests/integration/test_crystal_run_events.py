from pathlib import Path

import pytest

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.events import EventType
from botcolosseo.envs.single_agent import SingleAgentTaskEnv
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.scenarios.splits import TaskKind


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_idle_pickup_trace_has_no_false_objective_events() -> None:
    graph = RegionGraph.from_yaml(Path("assets/scenarios/crystal_run/src/regions.yaml"))
    env = SingleAgentTaskEnv(
        config_path=Path("assets/scenarios/crystal_run/crystal_run.cfg"),
        region_graph=graph,
    )
    forbidden = {EventType.PICKUP, EventType.SCORE, EventType.VALID_HIT}
    try:
        env.reset(seed=17, task=TaskKind.PICKUP)
        observed = []
        for _ in range(5):
            observed.extend(env.step(MacroAction.IDLE).events)
    finally:
        env.close()

    assert forbidden.isdisjoint(event.type for event in observed)


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_shooting_away_from_target_has_no_valid_hit() -> None:
    graph = RegionGraph.from_yaml(Path("assets/scenarios/crystal_run/src/regions.yaml"))
    env = SingleAgentTaskEnv(
        config_path=Path("assets/scenarios/crystal_run/crystal_run.cfg"),
        region_graph=graph,
    )
    try:
        env.reset(seed=17, task=TaskKind.STATIC_HIT)
        for _ in range(18):
            env.step(MacroAction.TURN_LEFT)
        events = env.step(MacroAction.ATTACK).events
    finally:
        env.close()

    assert EventType.VALID_HIT not in {event.type for event in events}
