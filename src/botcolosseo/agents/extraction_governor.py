from __future__ import annotations

import torch
from torch import nn

from botcolosseo.agents.model import ActorOutput


class AggressiveCapabilityGovernor(nn.Module):
    """Public-observation router from Aggressive behavior back to Strong Base."""

    def __init__(
        self,
        *,
        strong_base: nn.Module,
        aggressive: nn.Module,
        carried_value_threshold: int = 35,
        health_threshold: int = 40,
        remaining_time_threshold: float = 40.0,
    ) -> None:
        super().__init__()
        if not 0 < carried_value_threshold <= 150:
            raise ValueError("Aggressive carried-value threshold is invalid")
        if not 0 < health_threshold <= 100:
            raise ValueError("Aggressive health threshold is invalid")
        if not 0 < remaining_time_threshold <= 75:
            raise ValueError("Aggressive remaining-time threshold is invalid")
        if strong_base.hidden_size != aggressive.hidden_size:
            raise ValueError("Aggressive governor Actor hidden sizes do not match")
        self.strong_base = strong_base
        self.aggressive = aggressive
        self.hidden_size = strong_base.hidden_size
        self.carried_value_threshold = carried_value_threshold
        self.health_threshold = health_threshold
        self.remaining_time_threshold = remaining_time_threshold
        self.requires_grad_(False)

    def initial_state(
        self, batch_size: int, *, device: torch.device | str
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("Aggressive governor batch size must be positive")
        strong = self.strong_base.initial_state(batch_size, device=device)
        aggressive = self.aggressive.initial_state(batch_size, device=device)
        latch = torch.zeros(1, batch_size, 1, device=device)
        return torch.cat((strong, aggressive, latch), dim=-1)

    def forward(
        self,
        frames: torch.Tensor,
        scalars: torch.Tensor,
        previous_actions: torch.Tensor,
        masks: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> ActorOutput:
        batch = frames.shape[0]
        if hidden is None:
            hidden = self.initial_state(batch, device=frames.device)
        expected = self.hidden_size * 2 + 1
        if hidden.shape != (1, batch, expected):
            raise ValueError("Aggressive governor hidden state has wrong shape")
        strong_hidden = hidden[..., : self.hidden_size]
        aggressive_hidden = hidden[
            ..., self.hidden_size : self.hidden_size * 2
        ]
        latched = hidden[0, :, -1] > 0.5
        strong = self.strong_base(
            frames,
            scalars,
            previous_actions,
            masks,
            strong_hidden,
        )
        aggressive = self.aggressive(
            frames,
            scalars,
            previous_actions,
            masks,
            aggressive_hidden,
        )
        logits: list[torch.Tensor] = []
        for index in range(frames.shape[1]):
            trigger = (
                scalars[:, index, 2]
                >= self.carried_value_threshold / 150.0
            ) | (
                scalars[:, index, 0] <= self.health_threshold / 100.0
            ) | (
                scalars[:, index, 8]
                <= self.remaining_time_threshold / 75.0
            )
            latched = latched | trigger
            logits.append(
                torch.where(
                    latched.unsqueeze(-1),
                    strong.logits[:, index],
                    aggressive.logits[:, index],
                )
            )
        combined_hidden = torch.cat(
            (
                strong.hidden,
                aggressive.hidden,
                latched.reshape(1, batch, 1).to(strong.hidden.dtype),
            ),
            dim=-1,
        )
        return ActorOutput(
            torch.stack(logits, dim=1),
            aggressive.features,
            combined_hidden,
        )
