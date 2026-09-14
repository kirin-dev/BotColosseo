"""Command-conditioned fair executor; no privileged input or style override."""

from __future__ import annotations

import copy

import torch
from torch import nn

from botcolosseo.agents.model import ActorCriticOutput, ActorOutput, RecurrentActor
from botcolosseo.data.extraction_demonstrations import EXTRACTION_SCALAR_DIM
from botcolosseo.training.hierarchical_protocol import Command


class BoundedFiLM(nn.Module):
    def __init__(self, condition_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(condition_dim, 64), nn.Tanh(), nn.Linear(64, 2 * feature_dim)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.anchor_network = None

    def anchor_at_hard(self) -> None:
        """Freeze the original difficulty=1 transform; learn relative modulation."""
        if self.network[0].in_features != 1 or self.anchor_network is not None:
            raise ValueError("Hard anchoring requires an unanchored scalar difficulty FiLM")
        self.anchor_network = copy.deepcopy(self.network).requires_grad_(False)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Preserve legacy checkpoints and standard executor loading in every CLI.
        anchored = any(key.startswith(prefix + "anchor_network.") for key in state_dict)
        if anchored and self.anchor_network is None:
            self.anchor_at_hard()
        elif not anchored:
            self.anchor_network = None
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = (0.25 * self.network(condition).tanh()).chunk(2, dim=-1)
        if self.anchor_network is not None:
            hard = torch.ones_like(condition)
            base_scale, base_shift = (0.25 * self.anchor_network(hard).tanh()).chunk(2, -1)
            hard_scale, hard_shift = (0.25 * self.network(hard).tanh()).chunk(2, -1)
            baseline = features * (1 + base_scale) + base_shift
            controlled = baseline + features * (scale - hard_scale) + (shift - hard_shift)
            return torch.where(condition == 1, baseline, controlled)
        return features * (1 + scale) + shift


class CommandExecutor(RecurrentActor):
    def __init__(self, *, hidden_size: int = 256) -> None:
        super().__init__(scalar_dim=EXTRACTION_SCALAR_DIM, hidden_size=hidden_size)
        self.command_embedding = nn.Embedding(len(Command), 32)
        self.command_input = BoundedFiLM(32, 320)
        self.command_output = BoundedFiLM(32, hidden_size)
        self.difficulty_input = BoundedFiLM(1, 320)
        self.difficulty_output = BoundedFiLM(1, hidden_size)

    def initialize_from(self, actor: RecurrentActor) -> None:
        """Copy only compatible skill modules, retaining zero-init conditioning."""
        if (actor.scalar_dim, actor.hidden_size, actor.action_count) != (
            self.scalar_dim,
            self.hidden_size,
            self.action_count,
        ):
            raise ValueError("Strong initialization architecture mismatch")
        for name in ("visual_encoder", "scalar_encoder", "recurrent", "policy"):
            getattr(self, name).load_state_dict(getattr(actor, name).state_dict())

    def forward(
        self,
        frames: torch.Tensor,
        scalars: torch.Tensor,
        previous_actions: torch.Tensor,
        masks: torch.Tensor,
        commands: torch.Tensor,
        difficulty: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> ActorOutput:
        self._validate_inputs(frames, scalars, previous_actions, masks, hidden)
        batch, time = frames.shape[:2]
        if commands.shape != (batch, time) or commands.dtype != torch.long:
            raise ValueError("commands must be long [batch,time]")
        if commands.device != frames.device or difficulty.device != frames.device:
            raise ValueError("Conditions must share actor device")
        if not bool(((commands >= 0) & (commands < len(Command))).all()):
            raise ValueError("Command outside schema")
        if difficulty.shape != (batch, time, 1) or not difficulty.is_floating_point():
            raise ValueError("difficulty must be floating [batch,time,1]")
        if not bool(((difficulty >= 0) & (difficulty <= 1)).all()):
            raise ValueError("difficulty outside [0,1]")
        pixels = frames.float() / 255 if frames.dtype == torch.uint8 else frames
        visual = self.visual_encoder(pixels.reshape(-1, 1, 84, 84)).reshape(batch, time, -1)
        previous = nn.functional.one_hot(previous_actions.long(), self.action_count).to(
            scalars.dtype
        )
        own = self.scalar_encoder(torch.cat((scalars, previous), -1))
        condition = self.command_embedding(commands)
        encoded = self.command_input(torch.cat((visual, own), -1), condition)
        encoded = self.difficulty_input(encoded, difficulty)
        current = (
            self.initial_state(batch, device=frames.device)[0] if hidden is None else hidden[0]
        )
        outputs = []
        for index in range(time):
            current = self.recurrent(encoded[:, index], current * masks[:, index, None])
            outputs.append(current)
        features = torch.stack(outputs, 1)
        policy_features = self.command_output(features, condition)
        policy_features = self.difficulty_output(policy_features, difficulty)
        return ActorOutput(self.policy(policy_features), features, current[None])


class StrategicActor(nn.Module):
    """Low-frequency conditional command actor consuming detached fair features.

    Caller supplies only own public scalars, previous command and its duration.
    Strategy identity is the snapshot, not a style or difficulty label.
    """

    def __init__(self, *, executor_feature_dim: int = 256, hidden_size: int = 128) -> None:
        super().__init__()
        self.feature_dim = executor_feature_dim
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(executor_feature_dim + EXTRACTION_SCALAR_DIM + len(Command) + 1, 128),
            nn.Tanh(),
        )
        self.style_input = BoundedFiLM(3, 128)
        self.difficulty_input = BoundedFiLM(1, 128)
        self.recurrent = nn.GRUCell(128, hidden_size)
        self.style_output = BoundedFiLM(3, hidden_size)
        self.difficulty_output = BoundedFiLM(1, hidden_size)
        self.policy = nn.Linear(hidden_size, len(Command))

    def forward(
        self,
        features: torch.Tensor,
        scalars: torch.Tensor,
        previous_commands: torch.Tensor,
        elapsed: torch.Tensor,
        style: torch.Tensor,
        difficulty: torch.Tensor,
        masks: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> ActorOutput:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError("Expected fair executor features [batch,time,feature_dim]")
        batch, time = features.shape[:2]
        shapes = (
            (scalars, (batch, time, EXTRACTION_SCALAR_DIM)),
            (previous_commands, (batch, time)),
            (elapsed, (batch, time, 1)),
            (style, (batch, time, 3)),
            (difficulty, (batch, time, 1)),
            (masks, (batch, time)),
        )
        if time == 0 or any(x.shape != shape or x.device != features.device for x, shape in shapes):
            raise ValueError("Strategic input shape/device mismatch")
        if previous_commands.dtype != torch.long or not bool(
            ((previous_commands >= 0) & (previous_commands < len(Command))).all()
        ):
            raise ValueError("Invalid previous command")
        for condition in (style, difficulty):
            if not condition.is_floating_point() or not bool(
                ((condition >= 0) & (condition <= 1)).all()
            ):
                raise ValueError("Conditions must be in [0,1]")
        if not bool(((masks == 0) | (masks == 1)).all()):
            raise ValueError("Invalid recurrent reset mask")
        if not bool(torch.isfinite(elapsed).all() and (elapsed >= 0).all()):
            raise ValueError("Invalid command duration")
        if hidden is not None and (
            hidden.shape != (1, batch, self.hidden_size) or hidden.device != features.device
        ):
            raise ValueError("Invalid strategic hidden state")
        commands = nn.functional.one_hot(previous_commands, len(Command)).to(scalars.dtype)
        encoded = self.encoder(torch.cat((features.detach(), scalars, commands, elapsed / 8), -1))
        encoded = self.difficulty_input(self.style_input(encoded, style), difficulty)
        current = features.new_zeros(batch, self.hidden_size) if hidden is None else hidden[0]
        outputs = []
        for index in range(time):
            current = self.recurrent(encoded[:, index], current * masks[:, index, None])
            outputs.append(current)
        recurrent = torch.stack(outputs, 1)
        controlled = self.difficulty_output(self.style_output(recurrent, style), difficulty)
        return ActorOutput(self.policy(controlled), recurrent, current[None])


class CommandActorCritic(nn.Module):
    """Privileged value baseline conditioned on the actual command and difficulty."""

    def __init__(self, actor: CommandExecutor, *, privileged_dim: int = 20) -> None:
        super().__init__()
        if privileged_dim <= 0:
            raise ValueError("privileged_dim must be positive")
        self.actor = actor
        self.privileged_dim = privileged_dim
        self.value = nn.Sequential(
            nn.Linear(actor.hidden_size + privileged_dim + len(Command) + 1, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        frames: torch.Tensor,
        scalars: torch.Tensor,
        previous_actions: torch.Tensor,
        masks: torch.Tensor,
        commands: torch.Tensor,
        difficulty: torch.Tensor,
        privileged: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> ActorCriticOutput:
        if privileged.shape != (*frames.shape[:2], self.privileged_dim):
            raise ValueError("Invalid command Critic privileged shape")
        if privileged.device != frames.device or not privileged.is_floating_point():
            raise ValueError("Invalid privileged device or dtype")
        if not bool(torch.isfinite(privileged).all()):
            raise ValueError("Nonfinite privileged state")
        output = self.actor(frames, scalars, previous_actions, masks, commands, difficulty, hidden)
        one_hot = nn.functional.one_hot(commands, len(Command)).to(privileged.dtype)
        values = self.value(
            torch.cat((output.features.detach(), privileged, one_hot, difficulty), -1)
        )
        return ActorCriticOutput(output.logits, values.squeeze(-1), output.hidden)
