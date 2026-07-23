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


class RoutedStyledActorCritic(nn.Module):
    """Three public-observation style branches over one frozen recurrent Actor."""

    route_count = 3

    def __init__(self, base: AsymmetricActorCritic, *, bottleneck: int = 32) -> None:
        super().__init__()
        self.base = base
        self.base.actor.requires_grad_(False)
        self.adapters = nn.ModuleList(
            ResidualStyleAdapter(base.actor.hidden_size, bottleneck)
            for _ in range(self.route_count)
        )
        self.policies = nn.ModuleList(
            copy.deepcopy(base.actor.policy) for _ in range(self.route_count)
        )

    @classmethod
    def from_base(
        cls, base: AsymmetricActorCritic, *, bottleneck: int = 32
    ) -> RoutedStyledActorCritic:
        return cls(copy.deepcopy(base), bottleneck=bottleneck)

    def forward(
        self,
        frames: torch.Tensor,
        scalars: torch.Tensor,
        previous_actions: torch.Tensor,
        masks: torch.Tensor,
        privileged: torch.Tensor,
        hidden: torch.Tensor | None = None,
        *,
        route_modes: torch.Tensor,
    ) -> StyleActorCriticOutput:
        if route_modes.shape != frames.shape[:2] or route_modes.dtype != torch.long:
            raise ValueError("route_modes must be a long tensor matching batch and time")
        if int(route_modes.min()) < -1 or int(route_modes.max()) >= self.route_count:
            raise ValueError("route_modes contains an unsupported Explorer branch")
        safe_modes = route_modes.clamp_min(0)
        actor = self.base.actor(frames, scalars, previous_actions, masks, hidden)
        branch_logits = torch.stack(
            [
                policy(adapter(actor.features))
                for adapter, policy in zip(self.adapters, self.policies, strict=True)
            ],
            dim=2,
        )
        gather_index = safe_modes[..., None, None].expand(
            *route_modes.shape, 1, branch_logits.shape[-1]
        )
        logits = branch_logits.gather(2, gather_index).squeeze(2)
        privileged_features = self.base.privileged_encoder(privileged)
        values = self.base.value(
            torch.cat((actor.features, privileged_features), dim=-1)
        ).squeeze(-1)
        return StyleActorCriticOutput(logits, values, actor.hidden, actor.logits)

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def public_actor(self) -> RoutedStyledPublicActor:
        return RoutedStyledPublicActor(
            self.base.actor,
            self.adapters,
            self.policies,
        )


class RoutedStyledPublicActor(nn.Module):
    def __init__(
        self, base_actor: nn.Module, adapters: nn.ModuleList, policies: nn.ModuleList
    ) -> None:
        super().__init__()
        if len(adapters) != RoutedStyledActorCritic.route_count or len(policies) != len(
            adapters
        ):
            raise ValueError("Explorer public Actor requires exactly three branches")
        self.base_actor = base_actor
        self.adapters = adapters
        self.policies = policies
        self.hidden_size = base_actor.hidden_size

    def initial_state(
        self, batch_size: int, *, device: torch.device | str
    ) -> torch.Tensor:
        return self.base_actor.initial_state(batch_size, device=device)

    def forward(
        self,
        frames: torch.Tensor,
        scalars: torch.Tensor,
        previous_actions: torch.Tensor,
        masks: torch.Tensor,
        hidden: torch.Tensor | None = None,
        *,
        route_modes: torch.Tensor,
    ) -> ActorOutput:
        if route_modes.shape != frames.shape[:2] or route_modes.dtype != torch.long:
            raise ValueError("route_modes must be a long tensor matching batch and time")
        output = self.base_actor(
            frames, scalars, previous_actions, masks, hidden
        )
        branch_logits = torch.stack(
            [
                policy(adapter(output.features))
                for adapter, policy in zip(self.adapters, self.policies, strict=True)
            ],
            dim=2,
        )
        gather_index = route_modes[..., None, None].expand(
            *route_modes.shape, 1, branch_logits.shape[-1]
        )
        logits = branch_logits.gather(2, gather_index).squeeze(2)
        return ActorOutput(logits, output.features, output.hidden)
