from dataclasses import fields

import numpy as np
import pytest

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.types import ActorObservation


def make_observation(frame: np.ndarray) -> ActorObservation:
    return ActorObservation(
        frame=frame,
        health=100.0,
        ammo=50.0,
        attack_ready=True,
        has_core=False,
        home_score=0,
        away_score=0,
        remaining_tics=700,
        previous_action=MacroAction.IDLE,
    )


def test_actor_observation_copies_and_freezes_valid_frame() -> None:
    source = np.zeros((84, 84), dtype=np.uint8)

    observation = make_observation(source)
    source[0, 0] = 255

    assert observation.frame.shape == (84, 84)
    assert observation.frame.dtype == np.uint8
    assert observation.frame[0, 0] == 0
    assert not observation.frame.flags.writeable


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((84, 84, 3), dtype=np.uint8),
        np.zeros((42, 42), dtype=np.uint8),
        np.zeros((84, 84), dtype=np.float32),
    ],
)
def test_actor_observation_rejects_wrong_frame_contract(frame: np.ndarray) -> None:
    with pytest.raises(ValueError, match="84x84 uint8 grayscale"):
        make_observation(frame)


def test_actor_observation_schema_excludes_privileged_fields() -> None:
    field_names = {field.name for field in fields(ActorObservation)}
    forbidden = {
        "x",
        "y",
        "angle",
        "region",
        "region_id",
        "target_x",
        "target_y",
        "depth",
        "labels",
        "automap",
    }

    assert field_names.isdisjoint(forbidden)
