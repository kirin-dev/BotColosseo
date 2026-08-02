from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

from botcolosseo.agents.checkpoint import CheckpointMetadata
from botcolosseo.agents.model import ActorCriticOutput, ActorOutput, RecurrentActor
from botcolosseo.agents.style_model import ResidualStyleAdapter, StyleActorCriticOutput
from botcolosseo.data.extraction_demonstrations import EXTRACTION_SCALAR_DIM
from botcolosseo.envs.actions import MacroAction

EXTRACTION_PRIVILEGED_DIM = 20
DEFENSIVE_ATTACK_ACTIONS = tuple(
    int(action)
    for action in (
        MacroAction.ATTACK,
        MacroAction.FORWARD_ATTACK,
        MacroAction.TURN_LEFT_ATTACK,
        MacroAction.TURN_RIGHT_ATTACK,
    )
)


def create_extraction_actor() -> RecurrentActor:
    return RecurrentActor(scalar_dim=EXTRACTION_SCALAR_DIM)


class ExtractionActorCritic(nn.Module):
    """Fair-observation Extraction Actor with a privileged training-only Critic."""

    def __init__(self) -> None:
        super().__init__()
        self.actor = create_extraction_actor()
        self.privileged_dim = EXTRACTION_PRIVILEGED_DIM
        self.privileged_encoder = nn.Sequential(
            nn.Linear(self.privileged_dim, 128),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(self.actor.hidden_size + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def initial_state(
        self, batch_size: int, *, device: torch.device | str
    ) -> torch.Tensor:
        return self.actor.initial_state(batch_size, device=device)

    def forward(
        self,
        frames: torch.Tensor,
        scalars: torch.Tensor,
        previous_actions: torch.Tensor,
        masks: torch.Tensor,
        privileged: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> ActorCriticOutput:
        expected = (*frames.shape[:2], self.privileged_dim)
        if privileged.shape != expected or privileged.device != frames.device:
            raise ValueError("Extraction privileged input has the wrong shape or device")
        actor = self.actor(frames, scalars, previous_actions, masks, hidden)
        critic = self.privileged_encoder(privileged)
        values = self.value(
            torch.cat((actor.features.detach(), critic), dim=-1)
        ).squeeze(-1)
        return ActorCriticOutput(actor.logits, values, actor.hidden)


def create_extraction_actor_critic() -> ExtractionActorCritic:
    return ExtractionActorCritic()


def freeze_extraction_actor_backbone(model: ExtractionActorCritic) -> None:
    model.actor.visual_encoder.requires_grad_(False)
    model.actor.scalar_encoder.requires_grad_(False)
    model.actor.recurrent.requires_grad_(False)


def configure_extraction_actor_for_visual_curriculum(
    model: ExtractionActorCritic,
) -> tuple[nn.Parameter, ...]:
    """Freeze visual features except the final convolution used by the curriculum."""
    model.actor.visual_encoder.requires_grad_(False)
    final_convolution = model.actor.visual_encoder[4]
    if not isinstance(final_convolution, nn.Conv2d):
        raise TypeError("Extraction final visual curriculum layer is not Conv2d")
    final_convolution.requires_grad_(True)
    return tuple(final_convolution.parameters())


class ExtractionResidualStyleActorCritic(nn.Module):
    """Bounded learned delta logits over one frozen Extraction Strong Base."""

    def __init__(
        self,
        base: ExtractionActorCritic,
        *,
        bottleneck: int = 32,
        max_delta: float = 2.0,
    ) -> None:
        super().__init__()
        if bottleneck <= 0 or max_delta <= 0:
            raise ValueError("Extraction style dimensions must be positive")
        self.base = copy.deepcopy(base)
        self.base.requires_grad_(False)
        self.style_privileged_encoder = copy.deepcopy(base.privileged_encoder)
        self.style_value = copy.deepcopy(base.value)
        hidden = self.base.actor.hidden_size
        actions = self.base.actor.action_count
        self.delta_policy = nn.Sequential(
            nn.Linear(hidden, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, actions),
        )
        nn.init.zeros_(self.delta_policy[-1].weight)
        nn.init.zeros_(self.delta_policy[-1].bias)
        self.max_delta = float(max_delta)

    def initial_state(
        self, batch_size: int, *, device: torch.device | str
    ) -> torch.Tensor:
        return self.base.actor.initial_state(batch_size, device=device)

    def forward(
        self,
        frames: torch.Tensor,
        scalars: torch.Tensor,
        previous_actions: torch.Tensor,
        masks: torch.Tensor,
        privileged: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> StyleActorCriticOutput:
        actor = self.base.actor(
            frames,
            scalars,
            previous_actions,
            masks,
            hidden,
        )
        critic = self.style_privileged_encoder(privileged)
        values = self.style_value(
            torch.cat((actor.features, critic), dim=-1)
        ).squeeze(-1)
        delta = self.max_delta * torch.tanh(self.delta_policy(actor.features))
        return StyleActorCriticOutput(
            actor.logits + delta,
            values,
            actor.hidden,
            actor.logits,
        )

    def trainable_parameters(self):
        return (
            parameter for parameter in self.parameters() if parameter.requires_grad
        )

    def public_actor(self) -> ExtractionResidualStyleActor:
        return ExtractionResidualStyleActor(
            copy.deepcopy(self.base.actor),
            copy.deepcopy(self.delta_policy),
            max_delta=self.max_delta,
        )


class ExtractionResidualStyleActor(nn.Module):
    def __init__(
        self,
        base_actor: RecurrentActor,
        delta_policy: nn.Module,
        *,
        max_delta: float,
    ) -> None:
        super().__init__()
        self.base_actor = base_actor
        self.delta_policy = delta_policy
        self.max_delta = float(max_delta)
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
    ) -> ActorOutput:
        base = self.base_actor(
            frames,
            scalars,
            previous_actions,
            masks,
            hidden,
        )
        delta = self.max_delta * torch.tanh(self.delta_policy(base.features))
        return ActorOutput(
            base.logits + delta,
            base.features,
            base.hidden,
        )


class ExtractionDefensiveGuardedActor(nn.Module):
    """Block attack macros while carrying value under observable low-health risk."""

    def __init__(self, actor: ExtractionResidualStyleActor) -> None:
        super().__init__()
        self.actor = actor
        self.hidden_size = actor.hidden_size

    def initial_state(
        self, batch_size: int, *, device: torch.device | str
    ) -> torch.Tensor:
        return self.actor.initial_state(batch_size, device=device)

    def forward(
        self,
        frames: torch.Tensor,
        scalars: torch.Tensor,
        previous_actions: torch.Tensor,
        masks: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> ActorOutput:
        output = self.actor(frames, scalars, previous_actions, masks, hidden)
        low_health = scalars[..., 0] <= 0.40
        carrying_value = scalars[..., 2] >= (25 / 150)
        guarded = low_health & carrying_value
        attack_mask = torch.zeros_like(output.logits, dtype=torch.bool)
        attack_mask[..., DEFENSIVE_ATTACK_ACTIONS] = guarded.unsqueeze(-1)
        logits = output.logits.masked_fill(
            attack_mask,
            torch.finfo(output.logits.dtype).min,
        )
        return ActorOutput(logits, output.features, output.hidden)


class ExtractionResidualActor(nn.Module):
    """Small trainable style branch over a frozen Extraction Strong Base."""

    def __init__(self, base: RecurrentActor, *, bottleneck: int = 32) -> None:
        super().__init__()
        if base.scalar_dim != EXTRACTION_SCALAR_DIM:
            raise ValueError("Extraction base Actor has the wrong scalar dimension")
        self.base = copy.deepcopy(base)
        self.base.requires_grad_(False)
        self.adapter = ResidualStyleAdapter(base.hidden_size, bottleneck)
        self.policy = copy.deepcopy(base.policy)
        self.hidden_size = base.hidden_size

    def initial_state(
        self, batch_size: int, *, device: torch.device | str
    ) -> torch.Tensor:
        return self.base.initial_state(batch_size, device=device)

    def forward(
        self,
        frames: torch.Tensor,
        scalars: torch.Tensor,
        previous_actions: torch.Tensor,
        masks: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> ActorOutput:
        base = self.base(
            frames,
            scalars,
            previous_actions,
            masks,
            hidden,
        )
        features = self.adapter(base.features)
        return ActorOutput(self.policy(features), features, base.hidden)

    def trainable_parameters(self):
        return (
            parameter for parameter in self.parameters() if parameter.requires_grad
        )


def load_extraction_policy(
    checkpoint: Path,
    *,
    style: str,
    scenario_hash: str,
    device: torch.device,
) -> tuple[nn.Module, CheckpointMetadata]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported Extraction policy checkpoint")
    metadata = CheckpointMetadata(**payload["metadata"])
    if metadata.scenario_hash != scenario_hash:
        raise ValueError("Extraction policy scenario hash does not match")
    if style == "strong":
        model: nn.Module = create_extraction_actor()
    else:
        model = ExtractionResidualActor(create_extraction_actor())
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, metadata
