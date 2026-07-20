from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from botcolosseo.envs.video import write_mp4
from botcolosseo.envs.vizdoom_game import GameSettings, create_game


@dataclass(frozen=True)
class EpisodeSummary:
    index: int
    decisions: int
    total_reward: float
    terminated: bool
    first_frame_shape: tuple[int, ...]


@dataclass(frozen=True)
class SmokeSummary:
    episodes: tuple[EpisodeSummary, ...]
    all_terminated: bool
    video_path: str | None
    video_error: str | None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _one_hot_action(button_count: int, decision: int) -> list[float]:
    if button_count <= 0:
        raise RuntimeError("ViZDoom scenario exposes no available buttons")
    action = [0.0] * button_count
    action[decision % button_count] = 1.0
    return action


def run_smoke(
    settings: GameSettings,
    *,
    episodes: int = 2,
    max_decisions: int = 100,
    frame_skip: int = 4,
    video_path: Path | None = None,
    video_fps: int = 10,
    require_video: bool = False,
    game_builder: Callable[[GameSettings], Any] = create_game,
    video_writer: Callable[[list[np.ndarray], Path, int], Path] = write_mp4,
) -> SmokeSummary:
    if episodes <= 0 or max_decisions <= 0 or frame_skip <= 0:
        raise ValueError("episodes, max_decisions, and frame_skip must be positive")
    if require_video and video_path is None:
        raise ValueError("require_video=True requires video_path")

    game = game_builder(settings)
    episode_summaries: list[EpisodeSummary] = []
    recorded_frames: list[np.ndarray] = []
    video_error: str | None = None
    try:
        for episode_index in range(episodes):
            game.new_episode()
            decisions = 0
            first_frame_shape: tuple[int, ...] = ()
            while not game.is_episode_finished() and decisions < max_decisions:
                state = game.get_state()
                if state is None:
                    raise RuntimeError("ViZDoom returned no state before termination")
                frame = np.asarray(state.screen_buffer)
                if not first_frame_shape:
                    first_frame_shape = tuple(frame.shape)
                if video_path is not None and episode_index == 0:
                    recorded_frames.append(frame.copy())
                action = _one_hot_action(game.get_available_buttons_size(), decisions)
                game.make_action(action, frame_skip)
                decisions += 1
            terminated = game.is_episode_finished()
            episode_summaries.append(
                EpisodeSummary(
                    index=episode_index,
                    decisions=decisions,
                    total_reward=float(game.get_total_reward()),
                    terminated=terminated,
                    first_frame_shape=first_frame_shape,
                )
            )

        if video_path is not None:
            try:
                video_writer(recorded_frames, video_path, video_fps)
            except Exception as exc:
                video_error = str(exc)
                if require_video:
                    raise RuntimeError(f"Required smoke video failed: {exc}") from exc
    finally:
        game.close()

    return SmokeSummary(
        episodes=tuple(episode_summaries),
        all_terminated=all(item.terminated for item in episode_summaries),
        video_path=(
            str(video_path.resolve()) if video_path is not None and video_error is None else None
        ),
        video_error=video_error,
    )
