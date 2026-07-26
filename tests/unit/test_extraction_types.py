from __future__ import annotations

import numpy as np
import pytest

from botcolosseo.envs.extraction_types import (
    ExtractionActorObservation,
    normalized_extraction_progress,
    observation_health,
)


def test_extraction_observation_freezes_frame_and_validates_public_scalars() -> None:
    source = np.zeros((84, 84), dtype=np.uint8)

    observation = ExtractionActorObservation(
        frame=source,
        health=100,
        ammo=30,
        carried_value=35,
        free_slots=1,
        minimum_slot_value=10,
        banked_value=0,
        extraction_open=False,
        extraction_progress=0.0,
        remaining_time=75.0,
        previous_action=0,
    )
    source[0, 0] = 255

    assert observation.frame[0, 0] == 0
    assert observation.frame.flags.writeable is False


def test_extraction_helpers_hide_inactive_native_health() -> None:
    assert observation_health(1, 100) == 100
    assert observation_health(2, 100) == 0
    assert normalized_extraction_progress(105) == 1
    with pytest.raises(ValueError, match="ammo"):
        ExtractionActorObservation(
            frame=np.zeros((84, 84), dtype=np.uint8),
            health=100,
            ammo=41,
            carried_value=0,
            free_slots=3,
            minimum_slot_value=0,
            banked_value=0,
            extraction_open=False,
            extraction_progress=0.0,
            remaining_time=75.0,
            previous_action=0,
        )
