from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import torch

from botcolosseo.agents.hybrid_config import HybridPolicyConfig
from botcolosseo.agents.league_opponents import (
    OpponentSpec,
    _checkpoint_scenario_hash,
    sha256_file,
)
from botcolosseo.agents.model import AsymmetricActorCritic, RecurrentActor
from botcolosseo.agents.style_governor import (
    DefensiveGovernor,
    DefensiveGovernorConfig,
    ExplorerGovernor,
    ExplorerGovernorConfig,
    GovernorDecision,
    GovernorTelemetry,
    PublicStyleContext,
)
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_types import DuelActorObservation, DuelPrivilegedState
from botcolosseo.training.duel_rollout import actor_observation_tensors


class StyleGovernor(Protocol):
    def reset(self) -> None: ...

    def decide(self, context: PublicStyleContext) -> GovernorDecision: ...


def load_strong_base_actor(
    checkpoint: Path,
    *,
    checkpoint_sha256: str,
    scenario_hash: str,
    device: torch.device,
) -> RecurrentActor:
    path = checkpoint.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != checkpoint_sha256:
        raise ValueError("Strong Base checkpoint hash does not match")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported Strong Base checkpoint schema version")
    if _checkpoint_scenario_hash(payload) != scenario_hash:
        raise ValueError("Strong Base checkpoint scenario hash does not match")
    state_dict = payload.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError("Strong Base checkpoint is missing model weights")
    if any(name.startswith(("adapter.", "adapters.")) for name in state_dict):
        raise ValueError("Hybrid governor requires the learned Strong Base checkpoint")
    model = AsymmetricActorCritic()
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError("Strong Base checkpoint model dimensions do not match") from error
    actor = model.actor.to(device).eval()
    actor.requires_grad_(False)
    return actor


class HybridStylePolicy:
    def __init__(
        self,
        actor: RecurrentActor,
        governor: StyleGovernor,
        *,
        device: torch.device,
    ) -> None:
        self._actor = actor.to(device).eval()
        self._actor.requires_grad_(False)
        self._governor = governor
        self._device = device
        self._hidden: torch.Tensor | None = None
        self._episode_start = True
        self._decision_index = 0
        self._telemetry: list[GovernorTelemetry] = []

    @classmethod
    def load(
        cls,
        *,
        checkpoint: Path,
        checkpoint_sha256: str,
        scenario_hash: str,
        governor: StyleGovernor,
        device: torch.device,
    ) -> HybridStylePolicy:
        actor = load_strong_base_actor(
            checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            scenario_hash=scenario_hash,
            device=device,
        )
        return cls(actor, governor, device=device)

    @property
    def telemetry(self) -> tuple[GovernorTelemetry, ...]:
        return tuple(self._telemetry)

    def drain_telemetry(self) -> tuple[GovernorTelemetry, ...]:
        rows = self.telemetry
        self._telemetry.clear()
        return rows

    def reset(self) -> None:
        self._hidden = self._actor.initial_state(1, device=self._device)
        self._governor.reset()
        self._episode_start = True
        self._decision_index = 0
        self._telemetry.clear()

    @torch.inference_mode()
    def act(self, observation: DuelActorObservation) -> MacroAction:
        if self._hidden is None:
            raise RuntimeError("Hybrid policy must be reset before act")
        inputs = actor_observation_tensors(
            observation,
            episode_start=self._episode_start,
            device=self._device,
        )
        output = self._actor(*inputs, self._hidden)
        self._hidden = output.hidden
        self._episode_start = False
        logits = output.logits[0, 0]
        if logits.numel() != len(MacroAction) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("Strong Base emitted invalid action logits")
        base_action = MacroAction(int(logits.argmax()))
        context = PublicStyleContext(
            health=observation.health,
            armor=observation.armor,
            ammo=observation.ammo,
            own_score=observation.own_score,
            opponent_score=observation.opponent_score,
            has_core=observation.has_core,
            previous_action=MacroAction(observation.previous_action),
            base_logits=tuple(float(value) for value in logits.tolist()),
            decision_index=self._decision_index,
        )
        decision = self._governor.decide(context)
        final_action = self._resolve_action(logits, base_action, decision)
        self._telemetry.append(
            GovernorTelemetry(
                decision_index=self._decision_index,
                base_action=base_action,
                final_action=final_action,
                state=decision.state,
                trigger=decision.trigger,
                reason=decision.reason,
                intervened=decision.intervened,
                used_override=decision.override_action is not None,
                fallback_condition=decision.fallback_condition,
                route_mode=decision.route_mode,
            )
        )
        self._decision_index += 1
        return final_action

    def _resolve_action(
        self,
        logits: torch.Tensor,
        base_action: MacroAction,
        decision: GovernorDecision,
    ) -> MacroAction:
        if decision.override_action is not None:
            return MacroAction(decision.override_action)
        if not decision.intervened:
            return base_action
        bias = torch.tensor(decision.logit_bias, dtype=logits.dtype, device=logits.device)
        governed = logits + bias
        if not bool(torch.isfinite(governed).all()):
            return base_action
        return MacroAction(int(governed.argmax()))


class HybridEvaluationPolicy:
    def __init__(
        self,
        name: str,
        policy: HybridStylePolicy,
    ) -> None:
        if name not in ("defensive", "explorer"):
            raise ValueError("Hybrid evaluation policy name must be defensive or explorer")
        self.name = name
        self._policy = policy

    @property
    def telemetry(self) -> Sequence[GovernorTelemetry]:
        return self._policy.telemetry

    def drain_telemetry(self) -> tuple[GovernorTelemetry, ...]:
        return self._policy.drain_telemetry()

    def reset(self, *, seed: int) -> None:
        del seed
        self._policy.reset()

    def act(
        self,
        observation: DuelActorObservation,
        state: DuelPrivilegedState,
    ) -> MacroAction:
        del state
        return self._policy.act(observation)


def build_hybrid_evaluation_policy(
    config: HybridPolicyConfig,
    *,
    device: torch.device,
) -> HybridEvaluationPolicy:
    return HybridEvaluationPolicy(
        config.style,
        build_hybrid_style_policy(config, device=device),
    )


def build_hybrid_style_policy(
    config: HybridPolicyConfig,
    *,
    device: torch.device,
) -> HybridStylePolicy:
    if config.style == "defensive":
        if not isinstance(config.governor, DefensiveGovernorConfig):
            raise ValueError("Defensive hybrid config has the wrong governor type")
        governor: StyleGovernor = DefensiveGovernor(config.governor)
    else:
        if not isinstance(config.governor, ExplorerGovernorConfig):
            raise ValueError("Explorer hybrid config has the wrong governor type")
        governor = ExplorerGovernor(config.governor)
    return HybridStylePolicy.load(
        checkpoint=config.base_checkpoint,
        checkpoint_sha256=config.base_checkpoint_sha256,
        scenario_hash=config.scenario_hash,
        governor=governor,
        device=device,
    )


def checkpoint_spec_for_hybrid(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    scenario_hash: str,
    selection_evidence: str,
) -> OpponentSpec:
    return OpponentSpec(
        opponent_id="strong-base-hybrid-source",
        kind="checkpoint",
        checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_sha256,
        scenario_hash=scenario_hash,
        selection_evidence=selection_evidence,
    )
