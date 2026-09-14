"""Opponent curricula, distinct from the empirical game's payoff/meta-solver."""

from __future__ import annotations

import numpy as np


def response_distribution(
    equilibrium: np.ndarray,
    wins: np.ndarray,
    draws: np.ndarray,
    games: np.ndarray,
    *,
    completed_steps: int,
    budget_steps: int,
) -> np.ndarray:
    arrays = [np.asarray(x, dtype=np.float64) for x in (equilibrium, wins, draws, games)]
    sigma, wins, draws, games = arrays
    if sigma.ndim != 1 or not len(sigma) or any(x.shape != sigma.shape for x in arrays):
        raise ValueError("Opponent statistics need aligned nonempty vectors")
    if any(not np.isfinite(x).all() or (x < 0).any() for x in arrays):
        raise ValueError("Opponent statistics must be finite and nonnegative")
    if not np.isclose(sigma.sum(), 1, atol=1e-8, rtol=0) or (wins + draws > games).any():
        raise ValueError("Invalid probabilities or game counts")
    if budget_steps <= 0 or not 0 <= completed_steps <= budget_steps:
        raise ValueError("Invalid response budget progress")
    if completed_steps * 4 >= budget_steps * 3:
        return sigma.copy()
    # Beta(1,1) smoothing; draws contribute half a win, not a separate payoff.
    p = (wins + 0.5 * draws + 1) / (games + 2)
    weights = p * (1 - p) + 0.01
    return 0.7 * sigma + 0.3 * weights / weights.sum()
