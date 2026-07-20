from pathlib import Path

import numpy as np
import pytest
import torch

from botcolosseo.agents.model import ActorCriticOutput, AsymmetricActorCritic, RecurrentActor
from botcolosseo.envs.duel_protocol import DuelEvent, DuelEventType
from botcolosseo.envs.duel_types import (
    DuelActorObservation,
    DuelPrivilegedState,
    DuelStep,
)
from botcolosseo.envs.synchronous_duel import DuelObservations
from botcolosseo.scenarios.duel_splits import generate_duel_splits
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.training.curriculum import CurriculumPhase, OpponentCurriculum
from botcolosseo.training.duel_rollout import (
    DuelRolloutCollector,
    load_bc_actor_checkpoint,
    privileged_tensor,
)


def observation(*, score: int = 0, previous_action: int = 0) -> DuelActorObservation:
    return DuelActorObservation(
        frame=np.zeros((84, 84), dtype=np.uint8),
        health=100.0,
        armor=0.0,
        ammo=50.0,
        own_score=score,
        opponent_score=0,
        has_core=False,
        previous_action=previous_action,
    )


def state() -> DuelPrivilegedState:
    return DuelPrivilegedState(
        host_x=-640.0,
        host_y=10.0,
        host_angle=90.0,
        host_region="home",
        opponent_x=640.0,
        opponent_y=-10.0,
        opponent_angle=270.0,
        opponent_region="away",
        core_x=0.0,
        core_y=20.0,
        carrier=1,
        host_health=100.0,
        opponent_health=90.0,
        host_score=0,
        opponent_score=0,
        round_state=1,
        engine_tic=10,
    )


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(
        self, frames, scalars, previous_actions, masks, privileged, hidden
    ) -> ActorCriticOutput:
        del scalars, previous_actions
        batch, time = frames.shape[:2]
        current = hidden[0]
        outputs = []
        for index in range(time):
            current = current * masks[:, index, None] + 1.0 + self.anchor
            outputs.append(current)
        features = torch.stack(outputs, dim=1)
        logits = torch.zeros(batch, time, 13, device=frames.device) + self.anchor
        values = privileged[..., 0] * 0.0 + self.anchor
        return ActorCriticOutput(logits, values, features[:, -1].unsqueeze(0))


class FakeTeacher:
    def reset(self, *, seed: int) -> None:
        self.seed = seed

    def act(self, state: DuelPrivilegedState):
        del state
        return 0


class FakeEnv:
    def __init__(self, case) -> None:
        self.case = case
        self.steps = 0
        self.closed = False
        self.scales: list[float] = []
        self._state = state()

    def reset(self):
        return DuelObservations(observation(), observation()), object()

    def teacher_state(self) -> DuelPrivilegedState:
        return self._state

    def set_shaping_scale(self, scale: float) -> None:
        self.scales.append(scale)

    def step(self, host_action, opponent_action) -> DuelStep:
        del host_action, opponent_action
        self.steps += 1
        done = self.steps == 2
        host = observation(
            score=int(done and self.case.learner_side == "host"), previous_action=1
        )
        opponent = observation(
            score=int(done and self.case.learner_side == "opponent"), previous_action=1
        )
        event = (
            DuelEvent(
                DuelEventType.SCORE,
                self.case.learner_side,
                0,
                self.steps,
                self.steps * 4,
            ),
        ) if done else ()
        return DuelStep(
            host=host,
            opponent=opponent,
            host_reward=1.0,
            opponent_reward=-1.0,
            terminated=done,
            truncated=False,
            events=event,
            decision_index=self.steps,
            engine_tic=self.steps * 4,
            peer_tic_lag=0,
            pre_action_tics=0,
            action_tics=4,
        )

    def close(self) -> None:
        self.closed = True


def make_curriculum() -> OpponentCurriculum:
    cases = generate_duel_splits(master_seed=5, pairs_per_opponent=2)["train"]
    return OpponentCurriculum(
        cases,
        phases=(CurriculumPhase(0, ("random_legal", "fixed_route")),),
        shaping_decay_steps=10,
    )


def test_privileged_tensor_uses_learner_relative_order() -> None:
    host = privileged_tensor(state(), learner_side="host", device="cpu")
    opponent = privileged_tensor(state(), learner_side="opponent", device="cpu")

    assert host.shape == (1, 1, 12)
    assert host[0, 0, :4].tolist() == pytest.approx([-0.625, 0.015625, 0.0, 1.0])
    assert opponent[0, 0, :2].tolist() == pytest.approx([0.625, -0.015625])
    assert host[0, 0, -2:].tolist() == [1.0, 0.0]
    assert opponent[0, 0, -2:].tolist() == [0.0, 1.0]


def test_collector_resets_hidden_swaps_sides_and_accounts_episodes() -> None:
    created: list[FakeEnv] = []

    def environment_factory(case):
        environment = FakeEnv(case)
        created.append(environment)
        return environment

    collector = DuelRolloutCollector(
        FakeModel(),
        curriculum=make_curriculum(),
        graph=RegionGraph.from_yaml(Path("assets/scenarios/crystal_run/src/regions.yaml")),
        device=torch.device("cpu"),
        environment_factory=environment_factory,
        teacher_factory=lambda name, graph, side: FakeTeacher(),
        action_sampler=lambda distribution: distribution.logits.argmax(dim=-1),
    )
    try:
        collection = collector.collect(steps=5, start_environment_step=0)
    finally:
        collector.close()

    assert collection.environment_steps == 5
    assert len(collection.episodes) == 2
    assert collection.event_counts == {"learner:score": 2}
    assert collection.rollout.masks.tolist() == [[0.0, 1.0, 0.0, 1.0, 0.0]]
    assert collection.rollout.hidden[0, :, 0].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]
    assert (created[0].case.learner_side, created[1].case.learner_side) == (
        "host",
        "opponent",
    )
    assert created[0].case.seed == created[1].case.seed
    assert all(environment.closed for environment in created)
    assert created[0].scales == pytest.approx([1.0, 0.9])


def test_collector_closes_environment_on_failure() -> None:
    environment = FakeEnv(make_curriculum().case(0, 0))

    def fail(*args, **kwargs):
        raise RuntimeError("step failed")

    environment.step = fail
    collector = DuelRolloutCollector(
        FakeModel(),
        curriculum=make_curriculum(),
        graph=RegionGraph.from_yaml(Path("assets/scenarios/crystal_run/src/regions.yaml")),
        device=torch.device("cpu"),
        environment_factory=lambda case: environment,
        teacher_factory=lambda name, graph, side: FakeTeacher(),
    )

    with pytest.raises(RuntimeError, match="step failed"):
        collector.collect(steps=1, start_environment_step=0)

    assert environment.closed


def test_bc_actor_checkpoint_loads_actor_only_with_provenance(tmp_path: Path) -> None:
    torch.manual_seed(31)
    actor = RecurrentActor()
    path = tmp_path / "bc.pt"
    torch.save(
        {
            "schema_version": 1,
            "model": actor.state_dict(),
            "metadata": {
                "config_hash": "bc-config",
                "scenario_hash": "scenario",
                "counters": {"updates": 7},
            },
        },
        path,
    )
    model = AsymmetricActorCritic()
    critic_before = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if not name.startswith("actor.")
    }

    metadata = load_bc_actor_checkpoint(
        path, model, expected_scenario_hash="scenario"
    )

    assert metadata.counters == {"updates": 7}
    assert all(
        torch.equal(value, actor.state_dict()[name])
        for name, value in model.actor.state_dict().items()
    )
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in critic_before.items()
    )
    with pytest.raises(ValueError, match="scenario"):
        load_bc_actor_checkpoint(path, model, expected_scenario_hash="other")
