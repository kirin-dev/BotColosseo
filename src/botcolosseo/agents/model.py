from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class ActorOutput(NamedTuple):
    logits: torch.Tensor
    features: torch.Tensor
    hidden: torch.Tensor


class ActorCriticOutput(NamedTuple):
    logits: torch.Tensor
    values: torch.Tensor
    hidden: torch.Tensor


class RecurrentActor(nn.Module):
    def __init__(
        self,
        *,
        action_count: int = 13,
        scalar_dim: int = 6,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        if action_count <= 0 or scalar_dim <= 0 or hidden_size <= 0:
            raise ValueError("Actor dimensions must be positive")
        self.action_count = action_count
        self.scalar_dim = scalar_dim
        self.hidden_size = hidden_size
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim + action_count, 64),
            nn.ReLU(),
        )
        self.recurrent = nn.GRUCell(256 + 64, hidden_size)
        self.policy = nn.Linear(hidden_size, action_count)

    def initial_state(
        self, batch_size: int, *, device: torch.device | str
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(
        self,
        frames: torch.Tensor,
        scalars: torch.Tensor,
        previous_actions: torch.Tensor,
        masks: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> ActorOutput:
        if not torch.jit.is_tracing():
            self._validate_inputs(frames, scalars, previous_actions, masks, hidden)
        batch, time = frames.shape[:2]
        if hidden is None:
            hidden = self.initial_state(batch, device=frames.device)
        normalized_frames = frames.float()
        if frames.dtype == torch.uint8:
            normalized_frames = normalized_frames / 255.0
        visual = self.visual_encoder(normalized_frames.reshape(-1, 1, 84, 84))
        visual = visual.reshape(batch, time, -1)
        one_hot_actions = torch.nn.functional.one_hot(
            previous_actions.long(), num_classes=self.action_count
        ).to(scalars.dtype)
        scalar_features = self.scalar_encoder(
            torch.cat((scalars, one_hot_actions), dim=-1)
        )
        encoded = torch.cat((visual, scalar_features), dim=-1)
        current = hidden[0]
        outputs: list[torch.Tensor] = []
        for index in range(time):
            current = current * masks[:, index].unsqueeze(-1)
            current = self.recurrent(encoded[:, index], current)
            outputs.append(current)
        features = torch.stack(outputs, dim=1)
        return ActorOutput(self.policy(features), features, current.unsqueeze(0))

    def _validate_inputs(
        self,
        frames: torch.Tensor,
        scalars: torch.Tensor,
        previous_actions: torch.Tensor,
        masks: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> None:
        if frames.ndim != 5 or tuple(frames.shape[2:]) != (1, 84, 84):
            raise ValueError("frames must have shape [batch, time, 1, 84, 84]")
        batch, time = frames.shape[:2]
        if time <= 0:
            raise ValueError("Actor sequences must not be empty")
        if frames.dtype != torch.uint8 and not frames.is_floating_point():
            raise ValueError("frames must be uint8 or floating point")
        if not scalars.is_floating_point():
            raise ValueError("scalars must be floating point")
        if previous_actions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("previous_actions must use an integer dtype")
        if scalars.shape != (batch, time, self.scalar_dim):
            raise ValueError("scalars have the wrong shape")
        if previous_actions.shape != (batch, time):
            raise ValueError("previous_actions have the wrong shape")
        if masks.shape != (batch, time):
            raise ValueError("masks have the wrong shape")
        if previous_actions.numel() and (
            int(previous_actions.min()) < 0
            or int(previous_actions.max()) >= self.action_count
        ):
            raise ValueError("previous action is outside the action space")
        if not bool(torch.all((masks == 0) | (masks == 1))):
            raise ValueError("masks must contain only zero or one")
        if hidden is not None and hidden.shape != (1, batch, self.hidden_size):
            raise ValueError("hidden state has the wrong shape")
        if frames.device != scalars.device or frames.device != previous_actions.device:
            raise ValueError("Actor inputs must share a device")
        if masks.device != frames.device or (hidden is not None and hidden.device != frames.device):
            raise ValueError("Actor inputs must share a device")


class AsymmetricActorCritic(nn.Module):
    def __init__(self, *, privileged_dim: int = 12) -> None:
        super().__init__()
        if privileged_dim <= 0:
            raise ValueError("privileged_dim must be positive")
        self.privileged_dim = privileged_dim
        self.actor = RecurrentActor()
        self.privileged_encoder = nn.Sequential(
            nn.Linear(privileged_dim, 128),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        frames: torch.Tensor,
        scalars: torch.Tensor,
        previous_actions: torch.Tensor,
        masks: torch.Tensor,
        privileged: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> ActorCriticOutput:
        if privileged.shape != (*frames.shape[:2], self.privileged_dim):
            raise ValueError("privileged input has the wrong shape")
        if privileged.device != frames.device:
            raise ValueError("privileged input must share the Actor device")
        actor = self.actor(frames, scalars, previous_actions, masks, hidden)
        privileged_features = self.privileged_encoder(privileged)
        values = self.value(
            torch.cat((actor.features, privileged_features), dim=-1)
        ).squeeze(-1)
        return ActorCriticOutput(actor.logits, values, actor.hidden)
