from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from botcolosseo.envs.duel_worker import DuelWorker, DuelWorkerSettings, WorkerRole


class FakeState:
    screen_buffer = np.zeros((84, 84), dtype=np.uint8)


class FakeGame:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.closed = False
        self.time = 0

    def load_config(self, path: str) -> bool:
        self.calls.append(("load_config", path))
        return True

    def set_doom_map(self, value: str) -> None:
        self.calls.append(("map", value))

    def set_seed(self, value: int) -> None:
        self.calls.append(("seed", value))

    def set_window_visible(self, value: bool) -> None:
        self.calls.append(("visible", value))

    def set_sound_enabled(self, value: bool) -> None:
        self.calls.append(("sound", value))

    def set_audio_buffer_enabled(self, value: bool) -> None:
        self.calls.append(("audio", value))

    def set_screen_format(self, value) -> None:
        self.calls.append(("format", value))

    def set_mode(self, value) -> None:
        self.calls.append(("mode", value))

    def add_game_args(self, value: str) -> None:
        self.calls.append(("args", value))

    def init(self) -> None:
        self.calls.append(("init", None))

    def set_action(self, value: list[int]) -> None:
        self.calls.append(("action", value))

    def advance_action(self, tics: int, update_state: bool) -> None:
        assert tics == 1
        self.time += 1
        self.calls.append(("advance", update_state))

    def get_state(self):
        return FakeState()

    def get_episode_time(self) -> int:
        return self.time

    def get_server_state(self):
        return SimpleNamespace(tic=self.time)

    def is_episode_finished(self) -> bool:
        return False

    def is_player_dead(self) -> bool:
        return False

    def is_multiplayer_game(self) -> bool:
        return True

    def get_game_variable(self, variable) -> float:
        return 0.0

    def close(self) -> None:
        self.closed = True


def settings(tmp_path: Path, role: WorkerRole) -> DuelWorkerSettings:
    config = tmp_path / "duel.cfg"
    config.write_text("# fake\n", encoding="utf-8")
    return DuelWorkerSettings(role=role, config_path=config, seed=7, port=17432)


def test_host_and_opponent_receive_explicit_network_arguments(tmp_path: Path) -> None:
    host_game = FakeGame()
    opponent_game = FakeGame()
    host = DuelWorker(settings(tmp_path, WorkerRole.HOST), game_factory=lambda: host_game)
    opponent = DuelWorker(
        settings(tmp_path, WorkerRole.OPPONENT), game_factory=lambda: opponent_game
    )

    host("init", None)
    opponent("init", None)

    host_args = next(value for name, value in host_game.calls if name == "args")
    opponent_args = next(value for name, value in opponent_game.calls if name == "args")
    assert "-host 2" in host_args and "-port 17432" in host_args
    assert "-join 127.0.0.1:17432" in opponent_args
    assert "+sv_noautoaim 1" in host_args


def test_extraction_settings_select_map_protocol_and_terminal_death(
    tmp_path: Path,
) -> None:
    game = FakeGame()
    extraction = DuelWorkerSettings(
        role=WorkerRole.HOST,
        config_path=settings(tmp_path, WorkerRole.HOST).config_path,
        seed=7,
        port=17432,
        map_name="MAP01",
        protocol_user_indices=tuple(range(1, 54)),
        force_respawn=False,
        time_limit_minutes=1.5,
        deathmatch=False,
    )
    worker = DuelWorker(extraction, game_factory=lambda: game)

    result = worker("init", None)

    assert ("map", "MAP01") in game.calls
    args = next(value for name, value in game.calls if name == "args")
    assert "+sv_forcerespawn 0" in args
    assert "+timelimit 1.5" in args
    assert "-deathmatch" not in args
    assert "+teamdamage 1" in args
    assert len(result["protocol_values"]) == 53


def test_layout_variant_is_forwarded_to_both_network_peers(tmp_path: Path) -> None:
    for role in (WorkerRole.HOST, WorkerRole.OPPONENT):
        game = FakeGame()
        configured = replace(settings(tmp_path, role), layout_variant=37)
        DuelWorker(configured, game_factory=lambda current=game: current)("init", None)
        args = next(value for name, value in game.calls if name == "args")
        assert "+set bot_extraction_layout_variant 37" in args


def test_step_advances_exactly_one_tic_with_explicit_update_flag(tmp_path: Path) -> None:
    game = FakeGame()
    worker = DuelWorker(settings(tmp_path, WorkerRole.HOST), game_factory=lambda: game)
    worker("init", None)

    result = worker("step", {"action": 1, "update_state": True})

    advances = [value for name, value in game.calls if name == "advance"]
    assert advances == [True]
    assert result["episode_time"] == 1
    assert result["frame"].shape == (84, 84)


def test_invalid_order_and_close_are_safe(tmp_path: Path) -> None:
    game = FakeGame()
    worker = DuelWorker(settings(tmp_path, WorkerRole.HOST), game_factory=lambda: game)

    with pytest.raises(RuntimeError, match="initialized"):
        worker("step", {"action": 0, "frame_skip": 1})
    worker("close", None)
    worker("close", None)

    assert game.closed is False
