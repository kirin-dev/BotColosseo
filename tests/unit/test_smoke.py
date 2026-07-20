from pathlib import Path
from types import SimpleNamespace

import numpy as np

from botcolosseo.envs.smoke import run_smoke
from botcolosseo.envs.vizdoom_game import GameSettings


class FakeGame:
    def __init__(self) -> None:
        self.tic = 0
        self.closed = False

    def new_episode(self) -> None:
        self.tic = 0

    def is_episode_finished(self) -> bool:
        return self.tic >= 3

    def get_state(self):
        return SimpleNamespace(screen_buffer=np.zeros((84, 84), dtype=np.uint8))

    def get_available_buttons_size(self) -> int:
        return 2

    def make_action(self, action: list[float], frame_skip: int) -> float:
        assert sum(action) == 1.0
        self.tic += 1
        return 1.0

    def get_total_reward(self) -> float:
        return 3.0

    def close(self) -> None:
        self.closed = True


def test_run_smoke_terminates_and_closes_game(tmp_path: Path) -> None:
    fake = FakeGame()
    settings = GameSettings(config_path=tmp_path / "unused.cfg")

    summary = run_smoke(
        settings,
        episodes=2,
        max_decisions=5,
        frame_skip=4,
        game_builder=lambda unused: fake,
    )

    assert summary.all_terminated
    assert [episode.decisions for episode in summary.episodes] == [3, 3]
    assert fake.closed


def test_optional_video_failure_does_not_fail_smoke(tmp_path: Path) -> None:
    fake = FakeGame()
    settings = GameSettings(config_path=tmp_path / "unused.cfg")

    def failing_writer(frames, path, fps):
        raise RuntimeError("encoder unavailable")

    summary = run_smoke(
        settings,
        episodes=1,
        max_decisions=5,
        frame_skip=4,
        video_path=tmp_path / "smoke.mp4",
        require_video=False,
        game_builder=lambda unused: fake,
        video_writer=failing_writer,
    )

    assert summary.all_terminated
    assert summary.video_error == "encoder unavailable"
    assert fake.closed
