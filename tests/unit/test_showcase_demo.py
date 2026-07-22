from __future__ import annotations

import numpy as np
import pytest

from botcolosseo.demo.showcase import (
    compose_learner_frame,
    compose_showcase_comparison,
)
from botcolosseo.envs.duel_types import DuelActorObservation


def observation() -> DuelActorObservation:
    return DuelActorObservation(
        frame=np.full((84, 84), 80, dtype=np.uint8),
        health=100.0,
        armor=0.0,
        ammo=10.0,
        own_score=1,
        opponent_score=0,
        has_core=True,
        previous_action=0,
    )


def test_learner_frame_has_fixed_rgb_geometry() -> None:
    frame = compose_learner_frame(
        observation(), policy_label="Aggressive", event_label="VALID_HIT"
    )

    assert frame.shape == (300, 256, 3)
    assert frame.dtype == np.uint8
    assert frame[48:].mean() > 0


def test_comparison_aligns_unequal_streams() -> None:
    base = [np.full((300, 256, 3), 10, dtype=np.uint8)]
    aggressive = [
        np.full((300, 256, 3), value, dtype=np.uint8) for value in (20, 30)
    ]

    frames = compose_showcase_comparison(
        (("Strong Base", base), ("Aggressive", aggressive)),
        subtitle="VALIDATION | seed=7 | vs fixed_route | host",
    )

    assert len(frames) == 2
    assert frames[0].shape == (332, 512, 3)
    assert np.array_equal(frames[1][-300:, :256], base[0])


def test_comparison_rejects_empty_stream() -> None:
    valid = [np.zeros((300, 256, 3), dtype=np.uint8)]
    with pytest.raises(ValueError, match="empty"):
        compose_showcase_comparison(
            (("Strong Base", valid), ("Aggressive", [])),
            subtitle="validation",
        )
