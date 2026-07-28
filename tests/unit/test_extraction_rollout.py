from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from botcolosseo.agents.model import ActorCriticOutput
from botcolosseo.data.extraction_demonstrations import ExtractionCase
from botcolosseo.envs.extraction_protocol import ExtractionEvent, ExtractionEventType
from botcolosseo.envs.extraction_rules import LifeState
from botcolosseo.envs.extraction_types import (
    ExtractionActorObservation,
    ExtractionObservations,
    ExtractionPrivilegedState,
    ExtractionStep,
)
from botcolosseo.training.extraction_rollout import (
    ExtractionCaseSchedule,
    ExtractionRolloutCollector,
    PolicyExtractionOpponentController,
    RandomLegalExtractionOpponentController,
    extraction_privileged_tensor,
)


def observation(*, banked: int = 0, previous_action: int = 0):
    return ExtractionActorObservation(
        frame=np.zeros((84, 84), dtype=np.uint8),
        health=100,
        ammo=30,
        carried_value=10,
        free_slots=2,
        minimum_slot_value=10,
        banked_value=banked,
        extraction_open=False,
        extraction_progress=0,
        remaining_time=70,
        previous_action=previous_action,
    )


def state() -> ExtractionPrivilegedState:
    return ExtractionPrivilegedState(
        host_x=-320,
        host_y=160,
        host_angle=90,
        opponent_x=320,
        opponent_y=-160,
        opponent_angle=270,
        host_health=100,
        opponent_health=80,
        host_slots=(10, 0, 0),
        opponent_slots=(25, 0, 0),
        host_banked=0,
        opponent_banked=0,
        cache_owner=0,
        cache_slots=(50, 0, 0),
        cache_x=0,
        cache_y=0,
        world_loot_mask=3,
        round_state=1,
        winner=0,
        engine_tic=100,
    )


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def initial_state(self, batch_size: int, *, device):
        return torch.zeros(1, batch_size, 256, device=device)

    def forward(
        self, frames, scalars, previous_actions, masks, privileged, hidden
    ) -> ActorCriticOutput:
        del scalars, previous_actions
        batch, time = frames.shape[:2]
        current = hidden[0]
        outputs = []
        for index in range(time):
            current = current * masks[:, index, None] + 1 + self.anchor
            outputs.append(current)
        logits = torch.zeros(batch, time, 13) + self.anchor
        values = privileged[..., 0] * 0 + self.anchor
        features = torch.stack(outputs, 1)
        return ActorCriticOutput(logits, values, features[:, -1].unsqueeze(0))


class FakeOpponent:
    def __init__(self) -> None:
        self.previous_actions: list[int] = []

    def reset(self, *, seed: int) -> None:
        self.seed = seed

    def act(self, observation, privileged_state):
        self.previous_actions.append(observation.previous_action)
        assert callable(privileged_state)
        return 0


class FakeEnvironment:
    def __init__(self, assignment) -> None:
        self.assignment = assignment
        self.steps = 0
        self.closed = False
        self._state = state()
        self._life = int(LifeState.ACTIVE)
        self._banked = 0

    def reset(self):
        return ExtractionObservations(observation(), observation(previous_action=3)), object()

    def privileged_state(self):
        return self._state

    def protocol_snapshot(self):
        public = SimpleNamespace(life_state=self._life, banked_value=self._banked)
        return SimpleNamespace(public_state=lambda side: public)

    def step(self, host_action, opponent_action):
        del host_action, opponent_action
        self.steps += 1
        done = self.steps == 2
        if done:
            self._life = int(LifeState.EXTRACTED)
            self._banked = 10
        learner_side = self.assignment.case.learner_side
        emitted = (
            ExtractionEvent(
                type=ExtractionEventType.EXTRACTED,
                side=learner_side,
                count=1,
                value=10,
                episode_id=0,
                decision_index=self.steps,
                engine_tic=100 + self.steps,
            ),
        ) if done else ()
        return ExtractionStep(
            host=observation(
                banked=10 if done and learner_side == "host" else 0,
                previous_action=1,
            ),
            opponent=observation(
                banked=10 if done and learner_side == "opponent" else 0,
                previous_action=1,
            ),
            host_reward=1 if learner_side == "host" else -1,
            opponent_reward=1 if learner_side == "opponent" else -1,
            terminated=done,
            truncated=False,
            winner=1 if learner_side == "host" else 2,
            events=emitted,
            decision_index=self.steps,
            engine_tic=100 + self.steps,
            peer_tic_lag=0,
        )

    def close(self) -> None:
        self.closed = True


def schedule() -> ExtractionCaseSchedule:
    return ExtractionCaseSchedule(
        (
            ExtractionCase("train", 11, "host", "strong"),
            ExtractionCase("train", 12, "opponent", "aggressive"),
        ),
        shaping_decay_steps=10,
    )


def test_extraction_privileged_tensor_is_learner_relative() -> None:
    host = extraction_privileged_tensor(state(), learner_side="host", device="cpu")
    opponent = extraction_privileged_tensor(
        state(), learner_side="opponent", device="cpu"
    )

    assert host.shape == (1, 1, 20)
    assert host[0, 0, :2].tolist() == pytest.approx([-0.5, 1 / 3])
    assert opponent[0, 0, :2].tolist() == pytest.approx([0.5, -1 / 3])
    assert host[0, 0, 8:10].tolist() == pytest.approx([1.0, 0.8])
    assert opponent[0, 0, 8:10].tolist() == pytest.approx([0.8, 1.0])


def test_extraction_collector_resets_hidden_swaps_sides_and_records_outcome() -> None:
    created: list[FakeEnvironment] = []
    opponent = FakeOpponent()

    def environment_factory(assignment):
        environment = FakeEnvironment(assignment)
        created.append(environment)
        return environment

    collector = ExtractionRolloutCollector(
        FakeModel(),
        schedule=schedule(),
        device=torch.device("cpu"),
        environment_factory=environment_factory,
        opponent_factory=lambda assignment, side: opponent,
        action_sampler=lambda distribution: distribution.logits.argmax(-1),
        teacher_supervision=True,
    )
    try:
        collection = collector.collect(steps=5, start_environment_step=0)
    finally:
        collector.close()

    assert collection.environment_steps == 5
    assert len(collection.episodes) == 2
    assert all(episode.extracted and episode.won for episode in collection.episodes)
    assert collection.event_counts == {"learner:extracted": 2}
    assert collection.rollout.scalars.shape == (1, 5, 9)
    assert collection.rollout.privileged.shape == (1, 5, 20)
    assert collection.rollout.teacher_actions is not None
    assert collection.rollout.teacher_mask is not None
    assert collection.rollout.route_modes is not None
    assert collection.rollout.teacher_mask.tolist() == [[True] * 5]
    assert collection.rollout.route_modes.tolist() == [[-1] * 5]
    assert collection.rollout.masks.tolist() == [[0.0, 1.0, 0.0, 1.0, 0.0]]
    assert [item.assignment.case.learner_side for item in created] == [
        "host",
        "opponent",
        "host",
    ]
    assert all(item.closed for item in created)
    assert opponent.previous_actions == [3, 1, 0, 1, 3]


class PublicActor(torch.nn.Module):
    def initial_state(self, batch_size: int, *, device):
        return torch.zeros(1, batch_size, 256, device=device)

    def forward(self, frames, scalars, previous_actions, masks, hidden):
        del scalars, previous_actions, masks
        logits = torch.zeros(*frames.shape[:2], 13)
        logits[..., 4] = 1
        return SimpleNamespace(logits=logits, hidden=hidden + 1)


def test_checkpoint_opponent_never_reads_privileged_state() -> None:
    controller = PolicyExtractionOpponentController(
        PublicActor(),
        device=torch.device("cpu"),
    )
    controller.reset(seed=9)
    action = controller.act(
        observation(),
        lambda: pytest.fail("privileged state was accessed"),
    )
    assert action == 4


def test_random_legal_extraction_opponent_is_seeded_and_public_only() -> None:
    first = RandomLegalExtractionOpponentController()
    second = RandomLegalExtractionOpponentController()
    first.reset(seed=17)
    second.reset(seed=17)

    first_actions = [
        first.act(
            observation(),
            lambda: pytest.fail("privileged state was accessed"),
        )
        for _ in range(8)
    ]
    second_actions = [
        second.act(
            observation(),
            lambda: pytest.fail("privileged state was accessed"),
        )
        for _ in range(8)
    ]

    assert first_actions == second_actions
