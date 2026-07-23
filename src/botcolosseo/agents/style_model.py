from __future__ import annotations

import copy
from typing import NamedTuple

import torch
from torch import nn

from botcolosseo.agents.model import ActorOutput, AsymmetricActorCritic


class StyleActorCriticOutput(NamedTuple):
    logits: torch.Tensor
    values: torch.Tensor
    hidden: torch.Tensor
    base_logits: torch.Tensor


class ResidualStyleAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int) -> None:
        super().__init__()
        if hidden_size <= 0 or bottleneck <= 0:
            raise ValueError("Style adapter dimensions must be positive")
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, hidden_size),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.layers(features)


class StyledActorCritic(nn.Module):
    """A policy-only residual branch over a frozen fair-observation actor."""

    def __init__(self, base: AsymmetricActorCritic, *, bottleneck: int = 32) -> None:
        super().__init__()
        self.base = base
        self.base.actor.requires_grad_(False)
        self.adapter = ResidualStyleAdapter(base.actor.hidden_size, bottleneck)
        self.policy = copy.deepcopy(base.actor.policy)
        self.policy.requires_grad_(True)

    @classmethod
    def from_base(
        cls, base: AsymmetricActorCritic, *, bottleneck: int = 32
    ) -> StyledActorCritic:
        return cls(copy.deepcopy(base), bottleneck=bottleneck)

    def forward(
        self,
        frames: torch.Tensor,
        scalars: torch.Tensor,
        previous_actions: torch.Tensor,
        masks: torch.Tensor,
        privileged: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> StyleActorCriticOutput:
        actor = self.base.actor(frames, scalars, previous_actions, masks, hidden)
        styled_features = self.adapter(actor.features)
        privileged_features = self.base.privileged_encoder(privileged)
        values = self.base.value(
            torch.cat((actor.features, privileged_features), dim=-1)
        ).squeeze(-1)
        return StyleActorCriticOutput(
            self.policy(styled_features), values, actor.hidden, actor.logits
        )

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def public_actor(self) -> StyledPublicActor:
        return StyledPublicActor(self.base.actor, self.adapter, self.policy)


class StyledPublicActor(nn.Module):
    def __init__(self, base_actor: nn.Module, adapter: nn.Module, policy: nn.Module) -> None:
        super().__init__()
        self.base_actor = base_actor
        self.adapter = adapter
        self.policy = policy
        self.hidden_size = base_actor.hidden_size

    def initial_state(
        self, batch_size: int, *, device: torch.device | str
    ) -> torch.Tensor:
        return self.base_actor.initial_state(batch_size, device=device)

    def forward(self, *args, **kwargs) -> ActorOutput:
        output = self.base_actor(*args, **kwargs)
        return ActorOutput(
            self.policy(self.adapter(output.features)), output.features, output.hidden
        )
