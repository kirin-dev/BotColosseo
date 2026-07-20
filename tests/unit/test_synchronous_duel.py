from pathlib import Path

import numpy as np
import pytest

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.synchronous_duel import SynchronousDuelEnv
from botcolosseo.scenarios.regions import RegionGraph


def protocol(*, tic: int = 10, winner: int = 0, round_state: int = 1):
    return (
        2,
        tic,
        round_state,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        winner,
        0,
        0,
        0,
        0,
        0,
    )


def worker_state(
    *, tic: int = 10, x: float = -640.0, winner: int = 0, health: float = 100.0
):
    return {
        "frame": np.zeros((240, 320), dtype=np.uint8),
        "episode_time": tic,
        "finished": winner > 0,
        "dead": False,
        "multiplayer": True,
        "protocol_values": protocol(tic=tic, winner=winner, round_state=2 if winner else 1),
        "player_x": x,
        "player_y": 0.0,
        "player_angle": 0.0,
        "health": health,
        "armor": 0.0,
        "ammo": 50.0,
    }


class FakeClient:
    def __init__(
        self, role: str, log: list[tuple[str, str]], *, invalid_initial: bool = False
    ) -> None:
        self.role = role
        self.log = log
        self.closed = False
        self.time = 10
        self.invalid_initial = invalid_initial
        self.last_command = ""

    def submit(self, command: str, payload: object) -> int:
        self.log.append((self.role, f"submit:{command}"))
        self.last_command = command
        if command == "step":
            self.time += 1
        return self.time

    def receive(self, request_id: int):
        self.log.append((self.role, "receive"))
        x = -640.0 if self.role == "host" else 640.0
        health = -999900.0 if self.invalid_initial and self.last_command == "init" else 100.0
        return worker_state(tic=request_id, x=x, health=health)

    def close(self) -> None:
        self.closed = True

    def is_alive(self) -> bool:
        return not self.closed


def make_env(
    log: list[tuple[str, str]], *, invalid_initial: bool = False
) -> SynchronousDuelEnv:
    graph = RegionGraph.from_yaml(Path("assets/scenarios/crystal_run/src/regions.yaml"))
    return SynchronousDuelEnv(
        config_path=Path("assets/scenarios/crystal_run/crystal_run.cfg"),
        region_graph=graph,
        seed=7,
        client_factory=lambda settings: FakeClient(
            settings.role.value, log, invalid_initial=invalid_initial
        ),
        port_allocator=lambda: 17432,
    )


def test_reset_and_step_dispatch_both_sides_before_receive() -> None:
    log: list[tuple[str, str]] = []
    env = make_env(log)
    try:
        observations, info = env.reset()
        step = env.step(MacroAction.MOVE_FORWARD, MacroAction.TURN_LEFT)
    finally:
        env.close()

    assert observations.host.frame.shape == (84, 84)
    assert info.port == 17432
    step_log = log[4:]
    for index in range(0, len(step_log), 4):
        assert step_log[index : index + 2] == [
            ("host", "submit:step"),
            ("opponent", "submit:step"),
        ]
    assert step.engine_tic == 14
    assert not step.terminated and not step.truncated


def test_tic_mismatch_closes_environment() -> None:
    log: list[tuple[str, str]] = []
    env = make_env(log)
    env.reset()
    env._opponent.time += 1

    with pytest.raises(RuntimeError, match="tic mismatch"):
        env.step(MacroAction.IDLE, MacroAction.IDLE)

    assert env._host is None and env._opponent is None


def test_reset_barriers_until_both_players_have_valid_initial_state() -> None:
    log: list[tuple[str, str]] = []
    env = make_env(log, invalid_initial=True)
    try:
        observations, info = env.reset()
    finally:
        env.close()

    assert observations.host.health == 100.0
    assert observations.opponent.health == 100.0
    assert info.engine_tic == 11
    assert ("host", "submit:step") in log
