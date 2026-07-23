from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_protocol import DuelEventType
from botcolosseo.envs.duel_types import DuelActorObservation
from botcolosseo.envs.synchronous_duel import SynchronousDuelEnv
from botcolosseo.evaluation.m2 import (
    EvaluationPolicy,
    TeacherEvaluationPolicy,
    valid_action_tic_boundary,
)
from botcolosseo.evaluation.showcase import ShowcaseMetricEvidence, case_id
from botcolosseo.scenarios.duel_splits import DuelCase
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.training.league_rollout import PublicCheckpointPolicy


@dataclass(frozen=True)
class ShowcaseEvent:
    decision_index: int
    label: str


@dataclass(frozen=True)
class RecordedShowcaseEpisode:
    policy_id: str
    case: DuelCase
    frames: tuple[NDArray[np.uint8], ...]
    events: tuple[ShowcaseEvent, ...]
    decisions: int
    learner_score: int
    opponent_score: int
    objective_completed: bool
    terminated: bool
    truncated: bool
    peer_tic_lag_max: int
    protocol_inconsistent: bool
    action_tic_inconsistent: bool
    score_event_inconsistent: bool
    environment_attempts: int
    scenario_hash: str

    def to_record(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "case": self.case.to_dict(),
            "case_id": case_id(self.case),
            "events": [asdict(event) for event in self.events],
            "decisions": self.decisions,
            "learner_score": self.learner_score,
            "opponent_score": self.opponent_score,
            "objective_completed": self.objective_completed,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "peer_tic_lag_max": self.peer_tic_lag_max,
            "protocol_inconsistent": self.protocol_inconsistent,
            "action_tic_inconsistent": self.action_tic_inconsistent,
            "score_event_inconsistent": self.score_event_inconsistent,
            "environment_attempts": self.environment_attempts,
            "scenario_hash": self.scenario_hash,
        }


class CheckpointEvaluationPolicy:
    def __init__(self, name: str, policy: PublicCheckpointPolicy) -> None:
        self.name = name
        self._policy = policy

    def reset(self, *, seed: int) -> None:
        del seed
        self._policy.reset()

    def act(self, observation: DuelActorObservation, state: object) -> MacroAction:
        del state
        return MacroAction(self._policy.act(observation))


EnvironmentFactory = Callable[[DuelCase], SynchronousDuelEnv]


def render_metrics_card(
    evidence: ShowcaseMetricEvidence, output: Path
) -> Path:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    labels = (
        ("Base win rate", f"{evidence.base_win_rate:.1%}"),
        ("Aggressive style shift", f"{evidence.aggressive_style_delta:+.3f}"),
        ("Skill retention", f"{evidence.skill_retention:.1%}"),
        ("Episodes", f"{evidence.episodes:,}"),
    )
    figure = Figure(figsize=(12, 2.4), dpi=150, facecolor="#111827")
    FigureCanvasAgg(figure)
    try:
        for index, (label, value) in enumerate(labels):
            axis = figure.add_subplot(1, 4, index + 1)
            axis.set_facecolor("#111827")
            axis.axis("off")
            axis.text(
                0.5,
                0.60,
                value,
                ha="center",
                va="center",
                color="white",
                fontsize=22,
                fontweight="bold",
            )
            axis.text(
                0.5,
                0.30,
                label,
                ha="center",
                va="center",
                color="#cbd5e1",
                fontsize=10,
            )
        figure.savefig(temporary, facecolor=figure.get_facecolor())
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        figure.clear()
    return output


def record_showcase_episode(
    case: DuelCase,
    *,
    policy_id: str,
    policy_label: str,
    policy: EvaluationPolicy,
    graph: RegionGraph,
    config_path: Path,
    max_decisions: int,
    environment_factory: EnvironmentFactory | None = None,
) -> RecordedShowcaseEpisode:
    if case.split != "validation":
        raise ValueError("Showcase episodes must use frozen validation cases")
    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
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
    try:
        opponent_side = "opponent" if case.learner_side == "host" else "host"
        opponent = TeacherEvaluationPolicy(case.opponent, graph, side=opponent_side)
        frames: list[NDArray[np.uint8]] = []
        events: list[ShowcaseEvent] = []
        action_tic_inconsistent = False
        peer_tic_lag_max = 0
        score_event_counts: Counter[str] = Counter()
        decisions = 0
        terminated = False
        truncated = False
        observations, reset_info = environment.reset()
        policy.reset(seed=case.seed ^ 0xA5A5A5A5)
        opponent.reset(seed=case.seed ^ 0x5A5A5A5A)
        learner_observation = (
            observations.host if case.learner_side == "host" else observations.opponent
        )
        initial_score = learner_observation.own_score
        initial_scores = {
            "host": observations.host.own_score,
            "opponent": observations.opponent.own_score,
        }
        while not (terminated or truncated):
            teacher_state = environment.teacher_state()
            learner_observation = (
                observations.host
                if case.learner_side == "host"
                else observations.opponent
            )
            learner_action = policy.act(learner_observation, None)
            opponent_action = opponent.act(
                observations.opponent
                if case.learner_side == "host"
                else observations.host,
                teacher_state,
            )
            host_action, opponent_action = (
                (learner_action, opponent_action)
                if case.learner_side == "host"
                else (opponent_action, learner_action)
            )
            step = environment.step(host_action, opponent_action)
            observations = type(observations)(step.host, step.opponent)
            decisions += 1
            if decisions > max_decisions:
                raise RuntimeError("Showcase episode exceeded max_decisions")
            terminated, truncated = step.terminated, step.truncated
            peer_tic_lag_max = max(peer_tic_lag_max, step.peer_tic_lag)
            action_tic_inconsistent |= not valid_action_tic_boundary(
                step.action_tics,
                terminated=step.terminated,
                truncated=step.truncated,
            )
            score_event_counts.update(
                event.side
                for event in step.events
                if event.type is DuelEventType.SCORE
            )
            learner_events = tuple(
                event
                for event in step.events
                if event.side == case.learner_side
                and event.type
                in (
                    DuelEventType.PICKUP,
                    DuelEventType.VALID_HIT,
                    DuelEventType.DROP,
                    DuelEventType.SCORE,
                )
            )
            events.extend(
                ShowcaseEvent(event.decision_index, event.type.name)
                for event in learner_events
            )
            learner_observation = (
                observations.host
                if case.learner_side == "host"
                else observations.opponent
            )
            event_label = ",".join(event.type.name for event in learner_events) or "NONE"
            frames.append(
                compose_learner_frame(
                    learner_observation,
                    policy_label=policy_label,
                    event_label=event_label,
                )
            )
        learner_observation = (
            observations.host if case.learner_side == "host" else observations.opponent
        )
        final_scores = {
            "host": observations.host.own_score,
            "opponent": observations.opponent.own_score,
        }
        score_event_inconsistent = any(
            score_event_counts[side] != final_scores[side] - initial_scores[side]
            for side in ("host", "opponent")
        )
        protocol_inconsistent = (
            peer_tic_lag_max != 0
            or action_tic_inconsistent
            or score_event_inconsistent
        )
        return RecordedShowcaseEpisode(
            policy_id=policy_id,
            case=case,
            frames=tuple(frames),
            events=tuple(events),
            decisions=decisions,
            learner_score=learner_observation.own_score,
            opponent_score=learner_observation.opponent_score,
            objective_completed=learner_observation.own_score > initial_score,
            terminated=terminated,
            truncated=truncated,
            peer_tic_lag_max=peer_tic_lag_max,
            protocol_inconsistent=protocol_inconsistent,
            action_tic_inconsistent=action_tic_inconsistent,
            score_event_inconsistent=score_event_inconsistent,
            environment_attempts=1,
            scenario_hash=reset_info.scenario_hash,
        )
    finally:
        environment.close()


def compose_learner_frame(
    observation: DuelActorObservation,
    *,
    policy_label: str,
    event_label: str,
) -> NDArray[np.uint8]:
    if not policy_label or not event_label:
        raise ValueError("Showcase overlay labels must be non-empty")
    view = cv2.resize(observation.frame, (256, 252), interpolation=cv2.INTER_NEAREST)
    view = cv2.cvtColor(view, cv2.COLOR_GRAY2RGB)
    canvas = np.zeros((300, 256, 3), dtype=np.uint8)
    canvas[48:] = view
    cv2.putText(
        canvas,
        policy_label,
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    state = (
        f"score={observation.own_score}-{observation.opponent_score} "
        f"core={int(observation.has_core)} event={event_label}"
    )
    cv2.putText(
        canvas,
        state,
        (6, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


def compose_showcase_comparison(
    streams: Sequence[tuple[str, Sequence[NDArray[np.uint8]]]],
    *,
    subtitle: str,
) -> tuple[NDArray[np.uint8], ...]:
    if not 2 <= len(streams) <= 4:
        raise ValueError("Showcase comparison requires two to four streams")
    if any(not label or not frames for label, frames in streams):
        raise ValueError("Showcase comparison contains an empty stream")
    shape = np.asarray(streams[0][1][0]).shape
    if shape != (300, 256, 3) or any(
        np.asarray(frame).shape != shape
        for _, frames in streams
        for frame in frames
    ):
        raise ValueError("Showcase comparison streams have incompatible geometry")
    result = []
    for index in range(max(len(frames) for _, frames in streams)):
        panels = [
            np.array(frames[min(index, len(frames) - 1)], copy=True)
            for _, frames in streams
        ]
        if len(panels) == 2:
            comparison = np.concatenate(panels, axis=1)
        else:
            panels.extend(
                np.zeros(shape, dtype=np.uint8)
                for _ in range(4 - len(panels))
            )
            comparison = np.concatenate(
                (
                    np.concatenate(panels[:2], axis=1),
                    np.concatenate(panels[2:], axis=1),
                ),
                axis=0,
            )
        canvas = np.zeros((332, comparison.shape[1], 3), dtype=np.uint8)
        if comparison.shape[0] == 600:
            canvas = np.zeros((632, comparison.shape[1], 3), dtype=np.uint8)
        canvas[32:] = comparison
        cv2.putText(
            canvas,
            subtitle,
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        result.append(canvas)
    return tuple(result)
