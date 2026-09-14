import numpy as np
import pytest

from botcolosseo.training.hierarchical_sampling import response_distribution


def test_pfsp_curriculum_then_exact_equilibrium():
    sigma = np.array([1.0, 0.0, 0.0])
    wins, draws, games = np.array([0, 50, 100]), np.zeros(3), np.full(3, 100)
    early = response_distribution(sigma, wins, draws, games, completed_steps=0, budget_steps=100)
    assert np.isclose(early.sum(), 1)
    assert (early > 0).all()
    assert early[1] > early[2]
    late = response_distribution(sigma, wins, draws, games, completed_steps=75, budget_steps=100)
    np.testing.assert_array_equal(late, sigma)


def test_bad_statistics_rejected():
    with pytest.raises(ValueError):
        response_distribution(
            np.array([1.0]),
            np.array([2.0]),
            np.array([0.0]),
            np.array([1.0]),
            completed_steps=0,
            budget_steps=100,
        )
