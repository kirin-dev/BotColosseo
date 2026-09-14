from types import SimpleNamespace

import numpy as np
import pytest
import torch

from botcolosseo.agents.hierarchical_controller import HierarchicalController
from botcolosseo.agents.hierarchical_critic import StrategicActorCritic
from botcolosseo.agents.hierarchical_model import CommandExecutor, StrategicActor
from botcolosseo.demo.hierarchical_controls import ScheduledController
from botcolosseo.training.hierarchical_collection import collect_strategic_episode
from botcolosseo.training.hierarchical_protocol import ControlCondition


class TinyGame:
    def __init__(self, *, timeout):
        self.timeout = timeout
        self.index = 0
        self.actions = []

    def observation(self, side):
        banked = (
            0
            if self.timeout
            else (
                30
                if side == "host" and self.index >= 1
                else 60
                if side == "opponent" and self.index >= 3
                else 0
            )
        )
        return SimpleNamespace(
            frame=np.zeros((84, 84), dtype=np.uint8),
            previous_action=0,
            health=100,
            ammo=30,
            carried_value=30,
            free_slots=2,
            minimum_slot_value=0,
            banked_value=banked,
            extraction_open=True,
            extraction_progress=0,
            remaining_time=75 - self.index,
        )

    def state(self):
        return SimpleNamespace(
            host=self.observation("host"),
            opponent=self.observation("opponent"),
            terminated=self.index == 3 and not self.timeout,
            truncated=self.index == 3 and self.timeout,
        )

    def reset(self):
        self.index = 0
        return self.state(), SimpleNamespace(scenario_hash="test")

    def step(self, host, opponent):
        self.actions.append((host, opponent))
        self.index += 1
        return self.state()

    def privileged_state(self):
        return SimpleNamespace(
            host_banked=self.observation("host").banked_value,
            opponent_banked=self.observation("opponent").banked_value,
        )


@pytest.mark.parametrize("timeout", [False, True])
def test_collection_local_end_and_timeout_bootstrap(monkeypatch, timeout):
    torch.set_num_threads(1)
    monkeypatch.setattr(
        "botcolosseo.training.hierarchical_collection.extraction_privileged_tensor",
        lambda *args, **kwargs: torch.zeros(1, 1, 20),
    )
    executor = CommandExecutor()
    controllers = {
        side: HierarchicalController(executor, StrategicActor(), seed=index)
        for index, side in enumerate(("host", "opponent"))
    }
    model = StrategicActorCritic(controllers["host"].strategy)
    with torch.no_grad():
        for parameter in model.value.parameters():
            parameter.zero_()
        model.value[-1].bias.fill_(2)
    env = TinyGame(timeout=timeout)
    batch, report = collect_strategic_episode(
        env, controllers, model, learner_side="host", expected_scenario="test"
    )
    assert report["global_decisions"] == 3
    assert bool(batch["truncated"][0, -1]) == timeout
    assert bool(batch["terminated"][0, -1]) != timeout
    assert batch["next_values"][0, -1] == (2 if timeout else 0)
    if timeout:
        assert report["learner_decisions"] == 3
        assert report["payoffs"] == (0, 0)
    else:
        assert report["learner_decisions"] == 1
        assert batch["rewards"].sum() == pytest.approx(0.2)
        assert report["payoffs"] == pytest.approx((0.2, 0.4))
        assert [action[0] for action in env.actions[1:]] == [0, 0]


def test_timeout_bootstrap_uses_scheduled_condition_without_rewriting_samples(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(
        "botcolosseo.training.hierarchical_collection.extraction_privileged_tensor",
        lambda *args, **kwargs: torch.zeros(1, 1, 20),
    )
    executor = CommandExecutor()
    neutral, changed = ControlCondition(), ControlCondition(aggressive=1, difficulty=0.5)
    host = ScheduledController(
        executor, StrategicActor(), seed=1, schedule=[(0, neutral), (1, changed)]
    )
    other = HierarchicalController(executor, StrategicActor(), seed=2)
    model = StrategicActorCritic(host.strategy)
    calls = []
    handle = model.register_forward_pre_hook(
        lambda _, args, kwargs: calls.append(
            (kwargs["style"].clone(), kwargs["difficulty"].clone())
        ),
        with_kwargs=True,
    )
    batch, _ = collect_strategic_episode(
        TinyGame(timeout=True),
        {"host": host, "opponent": other},
        model,
        learner_side="host",
        expected_scenario="test",
    )
    handle.remove()
    assert calls[-1][0].tolist() == [[[1, 0, 0]]]
    assert calls[-1][1].item() == 0.5
    assert batch["style"][0, 0].tolist() == [0, 0, 0]
    assert batch["difficulty"][0, 0].item() == 1
