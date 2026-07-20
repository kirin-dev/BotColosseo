from dataclasses import fields

import numpy as np
import pytest

from botcolosseo.envs.duel_types import DuelActorObservation, DuelPrivilegedState


def test_duel_actor_schema_contains_only_public_fields_and_freezes_frame() -> None:
    frame = np.zeros((84, 84), dtype=np.uint8)
    observation = DuelActorObservation(
        frame=frame,
        health=100.0,
        armor=0.0,
        ammo=50.0,
        own_score=1,
        opponent_score=2,
        has_core=True,
        previous_action=3,
    )
    frame[:] = 255

    assert observation.frame.max() == 0
    assert observation.frame.flags.writeable is False
    actor_fields = {field.name for field in fields(DuelActorObservation)}
    forbidden = {"position", "angle", "region", "core_x", "core_y", "opponent_x"}
    assert not any(token in name for name in actor_fields for token in forbidden)
    assert {field.name for field in fields(DuelPrivilegedState)} >= {
        "host_x",
        "opponent_x",
        "core_x",
        "host_region",
    }


@pytest.mark.parametrize(
    "frame",
    (
        np.zeros((42, 42), dtype=np.uint8),
        np.zeros((84, 84), dtype=np.float32),
    ),
)
def test_duel_actor_rejects_invalid_frame(frame: np.ndarray) -> None:
    with pytest.raises(ValueError):
        DuelActorObservation(frame, 100, 0, 50, 0, 0, False, 0)
