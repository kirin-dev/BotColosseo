from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from botcolosseo.agents.teachers import create_teacher
from botcolosseo.envs.events import EpisodeEvent
from botcolosseo.envs.single_agent import SingleAgentTaskEnv
from botcolosseo.envs.video import write_mp4
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.scenarios.splits import TaskKind


@dataclass(frozen=True)
class TeacherEpisodeSummary:
    task: str
    teacher: str
    seed: int
    success: bool
    truncated: bool
    decisions: int
    total_reward: float
    event_counts: dict[str, int]
    event_types: tuple[str, ...]
    first_frame_shape: tuple[int, ...]
    scenario_hash: str
    video_path: str | None
    video_error: str | None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def run_teacher_episode(
    *,
    task: TaskKind,
    teacher_name: str,
    seed: int,
    video_path: Path | None = None,
    require_video: bool = False,
    max_decisions: int = 225,
) -> TeacherEpisodeSummary:
    if require_video and video_path is None:
        raise ValueError("require_video=True requires video_path")
    root = Path(__file__).resolve().parents[3]
    graph = RegionGraph.from_yaml(root / "assets/scenarios/crystal_run/src/regions.yaml")
    env = SingleAgentTaskEnv(
        config_path=root / "assets/scenarios/crystal_run/crystal_run.cfg",
        region_graph=graph,
        max_decisions=max_decisions,
    )
    teacher = create_teacher(teacher_name, graph)
    events: list[EpisodeEvent] = []
    frames: list[np.ndarray] = []
    total_reward = 0.0
    success = False
    truncated = False
    scenario_hash = ""
    first_frame_shape: tuple[int, ...] = ()
    decisions = 0
    video_error: str | None = None
    try:
        observation, info = env.reset(seed=seed, task=task)
        teacher.reset(seed=seed, task=task)
        scenario_hash = info.scenario_hash
        first_frame_shape = tuple(observation.frame.shape)
        if video_path is not None:
            frames.append(observation.frame.copy())
        while not success and not truncated:
            action = teacher.act(env.teacher_state())
            step = env.step(action)
            decisions += 1
            total_reward += step.reward
            events.extend(step.events)
            if video_path is not None:
                frames.append(step.observation.frame.copy())
            success = step.terminated
            truncated = step.truncated
    finally:
        env.close()

    if video_path is not None:
        try:
            write_mp4(frames, video_path, fps=10)
        except Exception as exc:
            video_error = str(exc)
            if require_video:
                raise RuntimeError(f"Required Crystal Run video failed: {exc}") from exc
    event_types = tuple(event.type.value for event in events)
    return TeacherEpisodeSummary(
        task=task.value,
        teacher=teacher_name,
        seed=seed,
        success=success,
        truncated=truncated,
        decisions=decisions,
        total_reward=total_reward,
        event_counts=dict(sorted(Counter(event_types).items())),
        event_types=event_types,
        first_frame_shape=first_frame_shape,
        scenario_hash=scenario_hash,
        video_path=(
            str(video_path.resolve())
            if video_path is not None and video_error is None
            else None
        ),
        video_error=video_error,
    )
