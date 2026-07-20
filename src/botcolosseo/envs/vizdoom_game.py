from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import vizdoom as vzd


@dataclass(frozen=True)
class GameSettings:
    config_path: Path
    seed: int = 0
    visible: bool = False
    screen_format: vzd.ScreenFormat = vzd.ScreenFormat.GRAY8


def create_game(
    settings: GameSettings,
    game_factory: Callable[[], vzd.DoomGame] = vzd.DoomGame,
) -> vzd.DoomGame:
    config_path = settings.config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"ViZDoom config does not exist: {config_path}")

    game = game_factory()
    try:
        if not game.load_config(str(config_path)):
            raise ValueError(f"ViZDoom rejected config: {config_path}")
        game.set_seed(settings.seed)
        game.set_window_visible(settings.visible)
        game.set_sound_enabled(False)
        game.set_audio_buffer_enabled(False)
        game.set_screen_format(settings.screen_format)
        game.set_mode(vzd.Mode.PLAYER)
        game.init()
    except BaseException:
        game.close()
        raise
    return game
