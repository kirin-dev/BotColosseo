from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from botcolosseo.envs.duel_protocol import DuelEventType
from botcolosseo.envs.synchronous_duel import SynchronousDuelEnv
from botcolosseo.evaluation.m2 import EvaluationPolicy, TeacherEvaluationPolicy
from botcolosseo.evaluation.sync_audit import compose_duel_frame
from botcolosseo.scenarios.duel_splits import DuelCase
from botcolosseo.scenarios.regions import RegionGraph


@dataclass(frozen=True)
class ShowcaseEpisode:
    policy: str
    frames: tuple[NDArray[np.uint8], ...]
    decisions: int
    learner_score: int
    opponent_score: int
    objective_completed: bool
    terminated: bool
    truncated: bool


class EnvironmentFactory(Protocol):
    def __call__(self, case: DuelCase) -> SynchronousDuelEnv: ...


def compose_policy_comparison(
    streams: dict[str, list[NDArray[np.uint8]]], *, subtitle: str
) -> list[NDArray[np.uint8]]:
    if not streams or any(not frames for frames in streams.values()):
        raise ValueError("Policy comparison cannot contain an empty frame stream")
    first = next(iter(streams.values()))[0]
    shape = np.asarray(first).shape
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError("Policy comparison frames must be RGB images")
    if any(np.asarray(frame).shape != shape for frames in streams.values() for frame in frames):
        raise ValueError("Policy comparison frames must share one shape")

    combined: list[NDArray[np.uint8]] = []
    frame_count = max(len(frames) for frames in streams.values())
    for index in range(frame_count):
        columns: list[NDArray[np.uint8]] = []
        for policy, frames in streams.items():
            frame = np.array(frames[min(index, len(frames) - 1)], copy=True)
            canvas = np.zeros((shape[0] + 20, shape[1], 3), dtype=np.uint8)
            canvas[20:] = frame
            cv2.putText(
                canvas,
                policy,
                (5, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            columns.append(canvas)
        comparison = np.concatenate(columns, axis=1)
        canvas = np.zeros(
            (comparison.shape[0] + 24, comparison.shape[1], 3), dtype=np.uint8
        )
        canvas[24:] = comparison
        cv2.putText(
            canvas,
            subtitle,
            (5, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        combined.append(canvas)
    return combined


def record_showcase_episode(
    case: DuelCase,
    *,
    policy: EvaluationPolicy,
    graph: RegionGraph,
    config_path: Path,
    max_decisions: int,
    environment_factory: EnvironmentFactory | None = None,
) -> ShowcaseEpisode:
    if case.split != "validation":
        raise ValueError("M2 showcase must use a frozen validation case")
    environment = (
        environment_factory(case)
        if environment_factory is not None
        else SynchronousDuelEnv(
            config_path=config_path,
            region_graph=graph,
            seed=case.seed,
            max_decisions=max_decisions,
        )
    )
    opponent_side = "opponent" if case.learner_side == "host" else "host"
    opponent = TeacherEvaluationPolicy(case.opponent, graph, side=opponent_side)
    frames: list[NDArray[np.uint8]] = []
    terminated = False
    truncated = False
    decisions = 0
    try:
        observations, _ = environment.reset()
        policy.reset(seed=case.seed ^ 0xA5A5A5A5)
        opponent.reset(seed=case.seed ^ 0x5A5A5A5A)
        initial_score = (
            observations.host.own_score
            if case.learner_side == "host"
            else observations.opponent.own_score
        )
        while not (terminated or truncated):
            state = environment.teacher_state()
            learner_observation = (
                observations.host
                if case.learner_side == "host"
                else observations.opponent
            )
            opposing_observation = (
                observations.opponent
                if case.learner_side == "host"
                else observations.host
            )
            learner_action = policy.act(learner_observation, state)
            opponent_action = opponent.act(opposing_observation, state)
            host_action, away_action = (
                (learner_action, opponent_action)
                if case.learner_side == "host"
                else (opponent_action, learner_action)
            )
            step = environment.step(host_action, away_action)
            observations = type(observations)(step.host, step.opponent)
            decisions += 1
            terminated, truncated = step.terminated, step.truncated
            event_label = ",".join(
                event.type.value
                for event in step.events
                if event.type
                in (DuelEventType.PICKUP, DuelEventType.DROP, DuelEventType.SCORE)
            ) or "none"
            frames.append(compose_duel_frame(step, event_label=event_label))
        learner = (
            observations.host
            if case.learner_side == "host"
            else observations.opponent
        )
        return ShowcaseEpisode(
            policy=policy.name,
            frames=tuple(frames),
            decisions=decisions,
            learner_score=learner.own_score,
            opponent_score=learner.opponent_score,
            objective_completed=learner.own_score > initial_score,
            terminated=terminated,
            truncated=truncated,
        )
    finally:
        environment.close()
