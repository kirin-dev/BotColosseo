from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

from botcolosseo.agents.checkpoint import CheckpointMetadata
from botcolosseo.agents.model import ActorOutput, RecurrentActor
from botcolosseo.agents.style_model import ResidualStyleAdapter
from botcolosseo.data.extraction_demonstrations import EXTRACTION_SCALAR_DIM


def create_extraction_actor() -> RecurrentActor:
    return RecurrentActor(scalar_dim=EXTRACTION_SCALAR_DIM)


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
