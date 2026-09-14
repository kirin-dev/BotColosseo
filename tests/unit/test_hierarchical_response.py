import numpy as np
import pytest

from botcolosseo.evaluation.hierarchical_response import response_gain


def test_fixed_mixture_layout_paired_gain():
    baseline = np.full((4, 2, 2), 0.3)
    candidate = np.tile([0.4, 0.5], (4, 1))
    result = response_gain(candidate, baseline, [0.25, 0.75], [0.8, 0.2], draws=100)
    assert result["gain"] == pytest.approx(0.12)
    assert result["gain_ci95"] == pytest.approx([0.12, 0.12])
    assert result["improvement_supported"]


def test_small_or_uncertain_gain_not_promoted():
    baseline = np.full((4, 1, 1), 0.3)
    assert not response_gain(np.full((4, 1), 0.31), baseline, [1], [1])["improvement_supported"]
    result = response_gain([[0.1], [0.2], [0.5], [0.8]], baseline, [1], [1])
    assert result["gain"] > 0.02 and not result["improvement_supported"]
    with pytest.raises(ValueError):
        response_gain([[0.4]], baseline[:1], [1], [1])
    with pytest.raises(ValueError):
        response_gain(np.full((4, 1), np.nan), baseline, [1], [1])
