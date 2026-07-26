from __future__ import annotations

import torch

from botcolosseo.envs.extraction_types import ExtractionActorObservation
from botcolosseo.training.extraction_bc import extraction_observation_tensors


class ExtractionCheckpointController:
    def __init__(self, model: torch.nn.Module, *, device: torch.device) -> None:
        self.model = model
        self.device = device
        self._hidden: torch.Tensor | None = None
        self._episode_start = True

    def reset(self) -> None:
        self._hidden = self.model.initial_state(1, device=self.device)
        self._episode_start = True

    @torch.no_grad()
    def act(self, observation: ExtractionActorObservation) -> int:
        if self._hidden is None:
            raise RuntimeError("Extraction checkpoint controller must be reset")
        output = self.model(
            *extraction_observation_tensors(
                observation,
                episode_start=self._episode_start,
                device=self.device,
            ),
            self._hidden,
        )
        self._hidden = output.hidden
        self._episode_start = False
        return int(output.logits[0, 0].argmax())
