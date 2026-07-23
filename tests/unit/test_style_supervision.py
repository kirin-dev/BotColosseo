from pathlib import Path

import numpy as np

from botcolosseo.envs.duel_types import DuelActorObservation, DuelPrivilegedState
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.training.style_supervision import (
    DefensiveStyleSupervisor,
    ExplorerStyleSupervisor,
)


def _graph() -> RegionGraph:
    return RegionGraph.from_yaml(
        Path("assets/scenarios/crystal_run/src/regions.yaml")
    )


def _observation(score: int = 0) -> DuelActorObservation:
    return DuelActorObservation(
        frame=np.zeros((84, 84), dtype=np.uint8),
        health=100.0,
        armor=0.0,
        ammo=10.0,
        own_score=score,
        opponent_score=0,
        has_core=False,
        previous_action=0,
    )


def _state(**changes: object) -> DuelPrivilegedState:
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
    values.update(changes)
    return DuelPrivilegedState(**values)  # type: ignore[arg-type]


def test_defensive_supervision_is_masked_by_frozen_risk_predicate() -> None:
    supervisor = DefensiveStyleSupervisor(
        _graph(), side="host", episode_index=99
    )
    supervisor.reset(seed=7, initial_score=0)

    assert not supervisor.label(_state(), _observation()).supervised
    assert supervisor.label(
        _state(carrier=2, opponent_x=-200.0), _observation()
    ).supervised


def test_explorer_mode_round_robins_and_advances_on_public_score() -> None:
    starts = []
    for episode_index in range(4):
        supervisor = ExplorerStyleSupervisor(
            _graph(), side="host", episode_index=episode_index
        )
        supervisor.reset(seed=episode_index * 101, initial_score=4)
        starts.append(supervisor.route_mode(_observation(4)))
        assert supervisor.route_mode(_observation(5)) == (episode_index + 1) % 3

    assert starts == [0, 1, 2, 0]
