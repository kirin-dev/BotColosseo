from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from botcolosseo.agents.hybrid_policy import (
    HybridEvaluationPolicy,
    HybridStylePolicy,
    load_strong_base_actor,
)
from botcolosseo.agents.style_governor import (
    ACTION_COUNT,
    ZERO_BIAS,
    GovernorDecision,
    PublicStyleContext,
)
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_types import DuelActorObservation


class StubActor(torch.nn.Module):
    def __init__(self, logits: tuple[float, ...]) -> None:
        super().__init__()
        self.register_buffer("_logits", torch.tensor(logits, dtype=torch.float32))

    def initial_state(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return torch.zeros((1, batch_size, 1), device=device)

    def forward(self, *args: object, **kwargs: object) -> SimpleNamespace:
        hidden = args[-1]
        return SimpleNamespace(
            logits=self._logits.reshape(1, 1, -1),
            hidden=hidden,
        )


class RecordingGovernor:
    def __init__(self, decision: GovernorDecision) -> None:
        self.decision = decision
        self.contexts: list[PublicStyleContext] = []
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def decide(self, context: PublicStyleContext) -> GovernorDecision:
        self.contexts.append(context)
        return self.decision


def _observation() -> DuelActorObservation:
    return DuelActorObservation(
        frame=np.zeros((84, 84), dtype=np.uint8),
        health=90.0,
        armor=5.0,
        ammo=12.0,
        own_score=1,
        opponent_score=0,
        has_core=False,
        previous_action=int(MacroAction.TURN_LEFT),
    )


def _decision(
    *,
    bias: tuple[float, ...] = ZERO_BIAS,
    override: MacroAction | None = None,
) -> GovernorDecision:
    return GovernorDecision(
        state="test",
        trigger="test",
        reason="test",
        logit_bias=bias,
        override_action=override,
        max_remaining_interventions=1,
        fallback_condition="test",
    )


def test_hybrid_policy_requires_reset_and_exactly_falls_back_to_base() -> None:
    logits = [0.0] * ACTION_COUNT
    logits[MacroAction.ATTACK] = 3.0
    governor = RecordingGovernor(_decision())
    policy = HybridStylePolicy(StubActor(tuple(logits)), governor, device=torch.device("cpu"))

    with pytest.raises(RuntimeError, match="reset"):
        policy.act(_observation())
    policy.reset()

    assert policy.act(_observation()) is MacroAction.ATTACK
    assert governor.resets == 1
    assert governor.contexts[0].health == 90.0
    assert governor.contexts[0].previous_action is MacroAction.TURN_LEFT
    assert policy.telemetry[0].base_action is MacroAction.ATTACK
    assert policy.telemetry[0].final_action is MacroAction.ATTACK
    assert not policy.telemetry[0].intervened


def test_hybrid_policy_applies_bias_and_bounded_override() -> None:
    logits = [0.0] * ACTION_COUNT
    logits[MacroAction.ATTACK] = 1.0
    bias = [0.0] * ACTION_COUNT
    bias[MacroAction.STRAFE_LEFT] = 2.0
    biased = HybridStylePolicy(
        StubActor(tuple(logits)),
        RecordingGovernor(_decision(bias=tuple(bias))),
        device=torch.device("cpu"),
    )
    biased.reset()

    assert biased.act(_observation()) is MacroAction.STRAFE_LEFT

    override = HybridStylePolicy(
        StubActor(tuple(logits)),
        RecordingGovernor(_decision(override=MacroAction.MOVE_BACKWARD)),
        device=torch.device("cpu"),
    )
    override.reset()

    assert override.act(_observation()) is MacroAction.MOVE_BACKWARD
    assert override.telemetry[0].used_override


def test_hybrid_policy_reset_clears_telemetry_and_replays_deterministically() -> None:
    logits = tuple(float(index) for index in range(ACTION_COUNT))
    policy = HybridStylePolicy(
        StubActor(logits),
        RecordingGovernor(_decision()),
        device=torch.device("cpu"),
    )
    policy.reset()
    first = policy.act(_observation())
    assert len(policy.telemetry) == 1
    assert len(policy.drain_telemetry()) == 1
    assert not policy.telemetry

    policy.reset()
    second = policy.act(_observation())

    assert first is second
    assert policy.telemetry[0].decision_index == 0


def test_evaluation_adapter_ignores_seed_and_rejects_mislabeling() -> None:
    policy = HybridStylePolicy(
        StubActor((0.0,) * ACTION_COUNT),
        RecordingGovernor(_decision()),
        device=torch.device("cpu"),
    )
    adapter = HybridEvaluationPolicy("defensive", policy)

    adapter.reset(seed=123)
    assert adapter.act(_observation(), object()) is MacroAction.IDLE  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="defensive or explorer"):
        HybridEvaluationPolicy("learned", policy)


def test_strong_base_loader_rejects_hash_before_deserialization(tmp_path) -> None:
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"not a checkpoint")

    with pytest.raises(ValueError, match="hash"):
        load_strong_base_actor(
            checkpoint,
            checkpoint_sha256="0" * 64,
            scenario_hash="scenario",
            device=torch.device("cpu"),
        )
