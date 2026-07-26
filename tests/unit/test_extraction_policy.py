from __future__ import annotations

import numpy as np
import torch

from botcolosseo.agents.extraction_model import create_extraction_actor
from botcolosseo.agents.extraction_policy import ExtractionCheckpointController
from botcolosseo.envs.extraction_types import ExtractionActorObservation


def observation(previous_action: int = 0) -> ExtractionActorObservation:
    return ExtractionActorObservation(
        frame=np.zeros((84, 84), dtype=np.uint8),
        health=100,
        ammo=30,
        carried_value=0,
        free_slots=3,
        minimum_slot_value=0,
        banked_value=0,
        extraction_open=False,
        extraction_progress=0,
        remaining_time=75,
        previous_action=previous_action,
    )


def test_checkpoint_controller_requires_reset_and_returns_valid_action() -> None:
    controller = ExtractionCheckpointController(
        create_extraction_actor(),
        device=torch.device("cpu"),
    )

    try:
        controller.act(observation())
    except RuntimeError as error:
        assert "must be reset" in str(error)
    else:
        raise AssertionError("Controller accepted an uninitialized episode")

    controller.reset()
    action = controller.act(observation())

    assert 0 <= action < 13
