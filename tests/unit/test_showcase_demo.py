from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from botcolosseo.demo.showcase import (
    CheckpointEvaluationPolicy,
    RecordedShowcaseEpisode,
    ShowcaseEvent,
    compose_learner_frame,
    compose_showcase_comparison,
    record_showcase_episode,
)
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_protocol import DuelEvent, DuelEventType
from botcolosseo.envs.duel_types import DuelActorObservation, DuelPrivilegedState, DuelStep
from botcolosseo.envs.synchronous_duel import DuelObservations, DuelResetInfo
from botcolosseo.scenarios.duel_splits import DuelCase
from botcolosseo.scenarios.regions import RegionGraph


def observation() -> DuelActorObservation:
    return DuelActorObservation(
        frame=np.full((84, 84), 80, dtype=np.uint8),
        health=100.0,
        armor=0.0,
        ammo=10.0,
        own_score=1,
        opponent_score=0,
        has_core=True,
        previous_action=0,
    )


def test_learner_frame_has_fixed_rgb_geometry() -> None:
    frame = compose_learner_frame(
        observation(), policy_label="Aggressive", event_label="VALID_HIT"
    )

    assert frame.shape == (300, 256, 3)
    assert frame.dtype == np.uint8
    assert frame[48:].mean() > 0


def test_comparison_aligns_unequal_streams() -> None:
    base = [np.full((300, 256, 3), 10, dtype=np.uint8)]
    aggressive = [
        np.full((300, 256, 3), value, dtype=np.uint8) for value in (20, 30)
    ]

    frames = compose_showcase_comparison(
        (("Strong Base", base), ("Aggressive", aggressive)),
        subtitle="VALIDATION | seed=7 | vs fixed_route | host",
    )

    assert len(frames) == 2
    assert frames[0].shape == (332, 512, 3)
    assert np.array_equal(frames[1][-300:, :256], base[0])


def test_comparison_rejects_empty_stream() -> None:
    valid = [np.zeros((300, 256, 3), dtype=np.uint8)]
    with pytest.raises(ValueError, match="empty"):
        compose_showcase_comparison(
            (("Strong Base", valid), ("Aggressive", [])),
            subtitle="validation",
        )


def test_recorded_episode_serializes_without_frames() -> None:
    episode = RecordedShowcaseEpisode(
        policy_id="strong_base",
        case=DuelCase("validation", 1, 7, "fixed_route", "host", 0, "flank"),
        frames=(np.zeros((300, 256, 3), dtype=np.uint8),),
        events=(ShowcaseEvent(0, "PICKUP"),),
        decisions=1,
        learner_score=1,
        opponent_score=0,
        objective_completed=True,
        terminated=True,
        truncated=False,
        peer_tic_lag_max=0,
        protocol_inconsistent=False,
        action_tic_inconsistent=False,
        score_event_inconsistent=False,
        environment_attempts=1,
        scenario_hash="a" * 64,
    )

    payload = episode.to_record()

    assert "frames" not in payload
    assert payload["case_id"] == "fixed_route:1:host"
    assert payload["events"] == [{"decision_index": 0, "label": "PICKUP"}]


def test_checkpoint_adapter_does_not_use_privileged_state() -> None:
    calls: list[str] = []

    class FakePolicy:
        def reset(self) -> None:
            calls.append("reset")

        def act(self, actor_observation):
            calls.append("act")
            return 0

    adapter = CheckpointEvaluationPolicy("strong_base", FakePolicy())
    adapter.reset(seed=7)
    result = adapter.act(observation(), object())

    assert int(result) == 0
    assert calls == ["reset", "act"]


def test_record_showcase_episode_captures_learner_events_and_protocol() -> None:
    case = DuelCase("validation", 1, 7, "fixed_route", "host", 0, "flank")
    initial = _episode_observation()
    scored = _episode_observation(own_score=1)
    fake_environment = _FakeShowcaseEnvironment(
        (
            DuelStep(
                host=initial,
                opponent=initial,
                host_reward=0.0,
                opponent_reward=0.0,
                terminated=False,
                truncated=False,
                events=(DuelEvent(DuelEventType.PICKUP, "host", 0, 0, 4),),
                decision_index=0,
                engine_tic=4,
                peer_tic_lag=0,
                pre_action_tics=0,
                action_tics=4,
            ),
            DuelStep(
                host=scored,
                opponent=initial,
                host_reward=1.0,
                opponent_reward=-1.0,
                terminated=True,
                truncated=False,
                events=(DuelEvent(DuelEventType.SCORE, "host", 0, 1, 8),),
                decision_index=1,
                engine_tic=8,
                peer_tic_lag=0,
                pre_action_tics=0,
                action_tics=4,
            ),
        )
    )

    episode = record_showcase_episode(
        case,
        policy_id="strong_base",
        policy_label="Strong Base",
        policy=_FixedPolicy(),
        graph=RegionGraph.from_yaml(Path("assets/scenarios/crystal_run/src/regions.yaml")),
        config_path=Path("assets/scenarios/crystal_run/crystal_run.cfg"),
        max_decisions=2,
        environment_factory=lambda unused_case: fake_environment,
    )

    assert episode.decisions == 2
    assert episode.objective_completed is True
    assert [event.label for event in episode.events] == ["PICKUP", "SCORE"]
    assert episode.peer_tic_lag_max == 0
    assert episode.action_tic_inconsistent is False
    assert episode.score_event_inconsistent is False
    assert fake_environment.closed is True


def _episode_observation(
    *, own_score: int = 0, opponent_score: int = 0
) -> DuelActorObservation:
    return DuelActorObservation(
        frame=np.zeros((84, 84), dtype=np.uint8),
        health=100.0,
        armor=0.0,
        ammo=10.0,
        own_score=own_score,
        opponent_score=opponent_score,
        has_core=False,
        previous_action=0,
    )


class _FixedPolicy:
    name = "strong_base"

    def reset(self, *, seed: int) -> None:
        del seed

    def act(
        self, actor_observation: DuelActorObservation, state: DuelPrivilegedState
    ) -> MacroAction:
        del actor_observation, state
        return MacroAction.IDLE


class _FakeShowcaseEnvironment:
    def __init__(self, steps: tuple[DuelStep, ...]) -> None:
        self._steps = iter(steps)
        self.closed = False

    def reset(self) -> tuple[DuelObservations, DuelResetInfo]:
        initial = _episode_observation()
        return (
            DuelObservations(initial, initial),
            DuelResetInfo(7, 0, 0, 0, 2, "a" * 64),
        )

    def teacher_state(self) -> DuelPrivilegedState:
        return DuelPrivilegedState(
            host_x=-1.0,
            host_y=0.0,
            host_angle=0.0,
            host_region="home",
            opponent_x=1.0,
            opponent_y=0.0,
            opponent_angle=180.0,
            opponent_region="away",
            core_x=0.0,
            core_y=0.0,
            carrier=0,
            host_health=100.0,
            opponent_health=100.0,
            host_score=0,
            opponent_score=0,
            round_state=1,
            engine_tic=0,
        )

    def step(self, host_action: MacroAction, opponent_action: MacroAction) -> DuelStep:
        del host_action, opponent_action
        return next(self._steps)

    def close(self) -> None:
        self.closed = True
