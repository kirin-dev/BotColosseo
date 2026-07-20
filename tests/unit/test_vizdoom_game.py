from pathlib import Path

import vizdoom as vzd

from botcolosseo.envs.vizdoom_game import GameSettings, create_game


class FakeGame:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.closed = False

    def load_config(self, path: str) -> bool:
        self.calls.append(("load_config", path))
        return True

    def set_seed(self, value: int) -> None:
        self.calls.append(("set_seed", value))

    def set_window_visible(self, value: bool) -> None:
        self.calls.append(("set_window_visible", value))

    def set_sound_enabled(self, value: bool) -> None:
        self.calls.append(("set_sound_enabled", value))

    def set_audio_buffer_enabled(self, value: bool) -> None:
        self.calls.append(("set_audio_buffer_enabled", value))

    def set_screen_format(self, value: vzd.ScreenFormat) -> None:
        self.calls.append(("set_screen_format", value))

    def set_mode(self, value: vzd.Mode) -> None:
        self.calls.append(("set_mode", value))

    def init(self) -> None:
        self.calls.append(("init", None))

    def close(self) -> None:
        self.closed = True


def test_create_game_applies_headless_silent_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "basic.cfg"
    config_path.write_text("doom_map = map01\n", encoding="utf-8")
    fake = FakeGame()
    settings = GameSettings(config_path=config_path, seed=7)

    result = create_game(settings, game_factory=lambda: fake)

    assert result is fake
    assert ("set_seed", 7) in fake.calls
    assert ("set_window_visible", False) in fake.calls
    assert ("set_sound_enabled", False) in fake.calls
    assert ("set_audio_buffer_enabled", False) in fake.calls
    assert fake.calls[-1] == ("init", None)
    assert not fake.closed
