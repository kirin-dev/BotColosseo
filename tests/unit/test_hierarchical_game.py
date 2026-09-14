import numpy as np
import pytest

from botcolosseo.training.hierarchical_game import (
    approximate_bimatrix,
    equilibrium_residual,
    solve_bimatrix,
)


def test_matching_pennies_uniform_is_equilibrium():
    a = np.array([[1, -1], [-1, 1]])
    result = equilibrium_residual(a, -a, np.array([0.5, 0.5]), np.array([0.5, 0.5]))
    assert result.maximum_regret == 0


def test_general_sum_payoffs_are_not_negatives():
    a = np.array([[3, 0], [0, 2]])
    b = np.array([[2, 0], [0, 3]])
    result = equilibrium_residual(a, b, np.array([1.0, 0]), np.array([1.0, 0]))
    assert result.host_value == 3 and result.opponent_value == 2
    assert result.maximum_regret == 0
    off = equilibrium_residual(a, b, np.array([1.0, 0]), np.array([0.0, 1]))
    assert off.host_regret == 2 and off.opponent_regret == 2


def test_invalid_probability_rejected():
    with pytest.raises(ValueError):
        equilibrium_residual(
            np.ones((2, 2)), np.ones((2, 2)), np.array([0.8, 0.8]), np.array([0.5, 0.5])
        )


def test_solver_matching_pennies_and_general_sum_welfare():
    a = np.array([[1, -1], [-1, 1]])
    result = solve_bimatrix(a, -a)
    np.testing.assert_allclose(result.host, [0.5, 0.5])
    np.testing.assert_allclose(result.opponent, [0.5, 0.5])
    a = np.array([[4, 0], [0, 2]])
    b = np.array([[3, 0], [0, 2]])
    result = solve_bimatrix(a, b)
    assert result.residual.maximum_regret <= 1e-6
    assert result.residual.host_value + result.residual.opponent_value == 7


def test_solver_degenerate_rectangular_game():
    result = solve_bimatrix(np.ones((2, 3)), np.ones((2, 3)))
    assert result.residual.maximum_regret == 0
    np.testing.assert_allclose(result.host, [0.5, 0.5], atol=1e-5)
    np.testing.assert_allclose(result.opponent, [1 / 3, 1 / 3, 1 / 3], atol=1e-5)


def test_approximate_fallback_reports_measured_regret():
    a = np.array([[1.0, -1.0], [-1.0, 1.0]])
    result = approximate_bimatrix(a, -a)
    assert result.residual.maximum_regret <= 1e-3
    assert result.method == "approximate-multistart-regret"
    assert not result.entropy_refinement_complete


def test_failed_entropy_refinement_is_not_hidden(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "botcolosseo.training.hierarchical_game.minimize",
        lambda *args, **kwargs: SimpleNamespace(success=False),
    )
    result = solve_bimatrix(np.ones((2, 2)), np.ones((2, 2)))
    assert result.residual.maximum_regret == 0
    assert not result.entropy_refinement_complete
