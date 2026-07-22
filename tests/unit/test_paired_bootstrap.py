import numpy as np
import pytest

from botcolosseo.evaluation.paired_bootstrap import paired_bootstrap_difference


def test_paired_bootstrap_is_deterministic_with_frozen_resample_count() -> None:
    strong = np.asarray([2.0, 1.0, 3.0, 0.0])
    baseline = np.asarray([0.0, 0.0, 1.0, -1.0])

    first = paired_bootstrap_difference(strong, baseline)
    second = paired_bootstrap_difference(strong, baseline)

    assert first == second
    assert first.seed == 20260721
    assert first.samples == 10_000
    assert first.confidence == 0.95
    assert first.estimate == pytest.approx(1.5)
    assert first.lower >= 0.75
    assert first.upper <= 2.0


def test_paired_bootstrap_rejects_missing_nonfinite_or_unpaired_values() -> None:
    with pytest.raises(ValueError, match="non-empty paired vectors"):
        paired_bootstrap_difference([], [])
    with pytest.raises(ValueError, match="same shape"):
        paired_bootstrap_difference([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap_difference([float("nan")], [0.0])
