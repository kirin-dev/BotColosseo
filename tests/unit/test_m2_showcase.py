from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from botcolosseo.cli.render_m2_showcase import load_showcase_case
from botcolosseo.demo.m2_showcase import compose_policy_comparison


def test_policy_comparison_aligns_and_pads_frame_streams() -> None:
    ppo = [np.full((20, 30, 3), value, dtype=np.uint8) for value in (10, 20)]
    bc = [np.full((20, 30, 3), 30, dtype=np.uint8)]

    frames = compose_policy_comparison(
        {"PPO": ppo, "BC": bc},
        subtitle="VALIDATION SHOWCASE",
    )

    assert len(frames) == 2
    assert frames[0].shape == (64, 60, 3)
    assert np.array_equal(frames[1][-20:, 30:], bc[0])


def test_policy_comparison_rejects_empty_stream() -> None:
    with pytest.raises(ValueError, match="empty"):
        compose_policy_comparison(
            {"PPO": [np.zeros((20, 30, 3), dtype=np.uint8)], "BC": []},
            subtitle="VALIDATION SHOWCASE",
        )


def test_showcase_case_loader_uses_the_frozen_validation_schedule() -> None:
    case = load_showcase_case(
        Path("configs/m2/validation.json"),
        0,
    )

    assert case.split == "validation"
    assert case.seed == 656_489_971
