from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import vizdoom as vzd

from botcolosseo.envs.actions import MacroAction, action_vector
from botcolosseo.envs.ipc import ProcessClient


class WorkerRole(str, Enum):
    HOST = "host"
    OPPONENT = "opponent"


@dataclass(frozen=True)
class DuelWorkerSettings:
    role: WorkerRole
    config_path: Path
    seed: int
    port: int
    timeout: float = 10.0

    def __post_init__(self) -> None:
        if not 1024 <= self.port <= 65535:
            raise ValueError(f"Invalid duel port: {self.port}")
        if not 0 <= self.seed <= 2**31 - 1:
            raise ValueError(f"Invalid duel seed: {self.seed}")
        if self.timeout <= 0:
            raise ValueError("Duel worker timeout must be positive")


_DUEL_VARIABLES = (vzd.GameVariable.USER1,) + tuple(
    getattr(vzd.GameVariable, f"USER{index}") for index in range(21, 45)
)


class DuelWorker:
    def __init__(
        self,
        settings: DuelWorkerSettings,
        *,
        game_factory: Callable[[], Any] = vzd.DoomGame,
    ) -> None:
        self._settings = settings
        self._game_factory = game_factory
        self._game: Any | None = None

    def __call__(self, command: str, payload: object) -> object:
        if command == "init":
            return self._init()
        if command == "reset":
            game = self._require_game()
            game.new_episode()
            return self._state(game)
        if command == "step":
            return self._step(payload)
        if command == "respawn":
            game = self._require_game()
            game.respawn_player()
            return self._state(game)
        if command == "close":
            self.close()
            return None
        raise ValueError(f"Unknown duel worker command: {command}")

    def _init(self) -> dict[str, object]:
        if self._game is not None:
            raise RuntimeError("Duel worker is already initialized")
        config_path = self._settings.config_path.expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Duel config does not exist: {config_path}")
        game = self._game_factory()
        self._game = game
        try:
            if not game.load_config(str(config_path)):
                raise ValueError(f"ViZDoom rejected config: {config_path}")
            game.set_doom_map("MAP07")
            game.set_seed(self._settings.seed)
            game.set_window_visible(False)
            game.set_sound_enabled(False)
            game.set_audio_buffer_enabled(False)
            game.set_screen_format(vzd.ScreenFormat.GRAY8)
            game.set_mode(vzd.Mode.PLAYER)
            game.add_game_args(self._network_args())
            game.init()
            state = self._state(game)
            if not state["multiplayer"]:
                raise RuntimeError("ViZDoom worker did not enter multiplayer mode")
            return state
        except BaseException:
            self.close()
            raise

    def _network_args(self) -> str:
        if self._settings.role is WorkerRole.HOST:
            return (
                f"-host 2 -port {self._settings.port} -deathmatch "
                "+timelimit 1.0 +sv_forcerespawn 1 +sv_noautoaim 1 "
                "+sv_respawnprotect 0 +viz_respawn_delay 1 "
                "+name BotHost +colorset 0"
            )
        return (
            f"-join 127.0.0.1:{self._settings.port} "
            "+name BotOpponent +colorset 3"
        )

    def _step(self, payload: object) -> dict[str, object]:
        game = self._require_game()
        if not isinstance(payload, dict):
            raise TypeError("Duel step payload must be a dictionary")
        action = MacroAction(int(payload["action"]))
        update_state = bool(payload["update_state"])
        game.set_action(action_vector(action))
        game.advance_action(1, update_state)
        return self._state(game)

    def _state(self, game: Any) -> dict[str, object]:
        state = game.get_state()
        frame = None if state is None else np.array(state.screen_buffer, copy=True)
        return {
            "frame": frame,
            "episode_time": int(game.get_episode_time()),
            "server_tic": int(game.get_server_state().tic),
            "finished": bool(game.is_episode_finished()),
            "dead": bool(game.is_player_dead()),
            "multiplayer": bool(game.is_multiplayer_game()),
            "protocol_values": tuple(
                int(game.get_game_variable(variable)) for variable in _DUEL_VARIABLES
            ),
            "player_x": float(game.get_game_variable(vzd.GameVariable.POSITION_X)),
            "player_y": float(game.get_game_variable(vzd.GameVariable.POSITION_Y)),
            "player_angle": float(game.get_game_variable(vzd.GameVariable.ANGLE)),
            "health": float(game.get_game_variable(vzd.GameVariable.HEALTH)),
            "armor": float(game.get_game_variable(vzd.GameVariable.ARMOR)),
            "ammo": float(
                game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
            ),
            "hitcount": int(game.get_game_variable(vzd.GameVariable.HITCOUNT)),
        }

    def _require_game(self) -> Any:
        if self._game is None:
            raise RuntimeError("Duel worker must be initialized before use")
        return self._game

    def close(self) -> None:
        if self._game is not None:
            self._game.close()
            self._game = None


def spawn_duel_worker(settings: DuelWorkerSettings) -> ProcessClient:
    return ProcessClient.start(
        DuelWorker(settings),
        name=f"botcolosseo-duel-{settings.role.value}",
        timeout=settings.timeout,
    )
