from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import torch

from botcolosseo.agents.model import ActorCriticOutput
from botcolosseo.envs.extraction_types import ExtractionActorObservation, ExtractionPrivilegedState
from botcolosseo.training.hierarchical_protocol import Command
from botcolosseo.training.hierarchical_rollout import collect_command_episode


class FakeModel:
    def __init__(self):
        self.actor = SimpleNamespace(initial_state=lambda *args, **kwargs: torch.zeros(1, 1, 2))

    def eval(self):
        pass

    def __call__(self, **kwargs):
        return ActorCriticOutput(
            torch.zeros(1, 1, 13), torch.full((1, 1), 7.0), torch.zeros(1, 1, 2)
        )


class FakeEnv:
    def __init__(self, truncate=False):
        self.truncate = truncate
        self.t = 0
        self.obs = ExtractionActorObservation(
            frame=np.zeros((84, 84), dtype=np.uint8),
            health=100,
            ammo=30,
            carried_value=0,
            free_slots=3,
            minimum_slot_value=0,
            banked_value=0,
            extraction_open=True,
            extraction_progress=0,
            remaining_time=30,
            previous_action=0,
        )
        data = {field.name: 0 for field in fields(ExtractionPrivilegedState)}
        data.update(
            host_health=100,
            opponent_health=100,
            host_slots=(0, 0, 0),
            opponent_slots=(0, 0, 0),
            cache_slots=(0, 0, 0),
            world_loot_mask=127,
        )
        self.state = ExtractionPrivilegedState(**data)

    def reset(self):
        return SimpleNamespace(host=self.obs, opponent=self.obs), SimpleNamespace(
            scenario_hash="test"
        )

    def privileged_state(self):
        return self.state

    def protocol_snapshot(self):
        return None  # Extraction command scoring does not inspect pickup ledger.

    def step(self, host, opponent):
        self.t += 1
        if not self.truncate:
            self.obs = replace(self.obs, health=0)
            self.state = replace(self.state, host_health=0)
        return SimpleNamespace(
            host=self.obs,
            opponent=self.obs,
            terminated=not self.truncate and self.t == 2,
            truncated=self.truncate,
        )


def collect(env):
    return collect_command_episode(
        FakeModel(),
        env,
        side="host",
        layout_variant=0,
        schedule=lambda t: Command.EXTRACT_NORTH,
        opponent=SimpleNamespace(act=lambda state: 0),
        device="cpu",
    )


def test_learner_death_stops_experience_but_not_global_game():
    env = FakeEnv()
    data, summary = collect(env)
    assert env.t == 2
    assert summary["learner_steps"] == 1
    assert data["terminal"].item()
    assert data["bootstrap"].item() == 0
    assert sum(c["steps"] for c in summary["command_counts"].values()) == 1
    assert summary["command_counts"]["EXTRACT_NORTH"] == {
        "steps": 1,
        "starts": 1,
        "applicable": 1,
        "successes": 0,
    }
    for key, value in summary["discounted_reward_components"].items():
        assert value == float(data[key].sum())


def test_external_truncation_retains_value_bootstrap():
    data, summary = collect(FakeEnv(truncate=True))
    assert summary["truncated"]
    assert not data["terminal"].item()
    assert data["bootstrap"].item() == 7
