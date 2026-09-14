"""Restricted empirical bimatrix diagnostics; not full-game exploitability."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import linprog, minimize


@dataclass(frozen=True)
class BimatrixSolution:
    host: np.ndarray
    opponent: np.ndarray
    residual: EquilibriumResidual
    method: str
    entropy_refinement_complete: bool = False


def _supported_mixture(
    payoff: np.ndarray, best_actions: tuple[int, ...], support: tuple[int, ...]
) -> tuple[np.ndarray, bool] | None:
    """Opponent mixture making nominated actions best responses (LP, not zero-sum)."""
    columns = payoff[:, support]
    width = len(support)
    equality = np.zeros((len(best_actions) + 1, width + 1))
    equality[0, :width] = 1
    for row, action in enumerate(best_actions, start=1):
        equality[row, :width] = columns[action]
        equality[row, -1] = -1
    rhs = np.zeros(len(best_actions) + 1)
    rhs[0] = 1
    objective = np.zeros(width + 1)
    objective[-1] = -1  # Maximize this role's equilibrium payoff within support.
    result = linprog(
        objective,
        A_ub=np.column_stack((columns, -np.ones(len(columns)))),
        b_ub=np.zeros(len(columns)),
        A_eq=equality,
        b_eq=rhs,
        bounds=[(0, 1)] * width + [(None, None)],
        method="highs",
    )
    if not result.success:
        return None
    # Maximize entropy on the payoff-optimal feasible face. Remove redundant
    # equality rows (common in degenerate games) before constrained optimization.
    optimal = np.zeros(width + 1)
    optimal[-1] = 1
    full_eq = np.vstack((equality, optimal))
    full_rhs = np.r_[rhs, result.x[-1]]
    u, singular, _ = np.linalg.svd(full_eq, full_matrices=False)
    rank = int((singular > 1e-10 * max(1.0, singular[0])).sum())
    reduced_eq = u[:, :rank].T @ full_eq
    reduced_rhs = u[:, :rank].T @ full_rhs
    constraints = [
        {"type": "eq", "fun": lambda x: reduced_eq @ x - reduced_rhs},
    ]
    outside = [i for i in range(len(columns)) if i not in best_actions]
    if outside:
        constraints.append({"type": "ineq", "fun": lambda x: x[-1] - columns[outside] @ x[:width]})
    refined = minimize(
        lambda x: float((x[:width] * np.log(np.maximum(x[:width], 1e-15))).sum()),
        result.x,
        method="SLSQP",
        bounds=[(0, 1)] * width + [(None, None)],
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    entropy_ok = (
        refined.success
        and np.isfinite(refined.x).all()
        and np.max(np.abs(full_eq @ refined.x - full_rhs)) <= 1e-7
        and np.max(columns @ refined.x[:width] - refined.x[-1]) <= 1e-7
    )
    if entropy_ok:
        result.x = refined.x
    mixture = np.zeros(payoff.shape[1])
    mixture[list(support)] = np.maximum(result.x[:width], 0)
    if mixture.sum() <= 0:
        return None
    return mixture / mixture.sum(), bool(entropy_ok)


def solve_bimatrix(a: np.ndarray, b: np.ndarray) -> BimatrixSolution:
    """Enumerate small supports and verify every candidate against the full game.

    Degenerate supports are feasible LPs; no inversion of singular indifference
    equations. Each support is entropy-refined on its payoff-optimal face.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 2 or a.shape != b.shape or not all(1 <= n <= 6 for n in a.shape):
        raise ValueError("Solver supports role populations of size1..6")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Payoffs must be finite")
    candidates = []
    m, n = a.shape
    for size_h in range(1, m + 1):
        for h in combinations(range(m), size_h):
            for size_o in range(1, n + 1):
                for o in combinations(range(n), size_o):
                    q = _supported_mixture(a, h, o)
                    if q is None:
                        continue
                    p = _supported_mixture(b.T, o, h)
                    if p is None:
                        continue
                    p, p_refined = p
                    q, q_refined = q
                    residual = equilibrium_residual(a, b, p, q)
                    if residual.maximum_regret <= 1e-6:
                        entropy = -sum(float((x[x > 0] * np.log(x[x > 0])).sum()) for x in (p, q))
                        candidates.append(
                            (
                                residual.host_value + residual.opponent_value,
                                entropy,
                                p,
                                q,
                                residual,
                                p_refined and q_refined,
                            )
                        )
    if not candidates:
        return approximate_bimatrix(a, b)
    winner = max(candidates, key=lambda c: (round(c[0], 10), round(c[1], 10)))
    tie_complete = all(c[5] for c in candidates if abs(c[0] - winner[0]) <= 1e-8)
    return BimatrixSolution(winner[2], winner[3], winner[4], "support-enumeration-lp", tie_complete)


def approximate_bimatrix(a: np.ndarray, b: np.ndarray) -> BimatrixSolution:
    """Deterministic multistart maximum-regret minimization, accepted only <=1e-3."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 2 or a.shape != b.shape or not all(1 <= n <= 6 for n in a.shape):
        raise ValueError("Invalid fallback game dimensions")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite game")
    m, n = a.shape
    rng = np.random.default_rng(1701)
    starts = [np.r_[np.full(m, 1 / m), np.full(n, 1 / n), 1.0]]
    starts += [np.r_[rng.dirichlet(np.ones(m)), rng.dirichlet(np.ones(n)), 1.0] for _ in range(12)]

    def deviations(x):
        p, q = x[:m], x[m : m + n]
        return np.r_[x[-1] - (a @ q - p @ a @ q), x[-1] - (p @ b - p @ b @ q)]

    constraints = [
        {"type": "eq", "fun": lambda x: x[:m].sum() - 1},
        {"type": "eq", "fun": lambda x: x[m : m + n].sum() - 1},
        {"type": "ineq", "fun": deviations},
    ]
    accepted = []
    for start in starts:
        result = minimize(
            lambda x: x[-1],
            start,
            method="SLSQP",
            bounds=[(0, 1)] * (m + n) + [(0, None)],
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if not np.isfinite(result.x).all():
            continue
        p, q = np.maximum(result.x[:m], 0), np.maximum(result.x[m : m + n], 0)
        if p.sum() <= 0 or q.sum() <= 0:
            continue
        p, q = p / p.sum(), q / q.sum()
        residual = equilibrium_residual(a, b, p, q)
        if residual.maximum_regret <= 1e-3:
            accepted.append(BimatrixSolution(p, q, residual, "approximate-multistart-regret"))
    if not accepted:
        raise RuntimeError("No approximate equilibrium meets regret <=1e-3")
    return min(
        accepted,
        key=lambda c: (
            c.residual.maximum_regret,
            -c.residual.host_value - c.residual.opponent_value,
        ),
    )


@dataclass(frozen=True)
class EquilibriumResidual:
    host_value: float
    opponent_value: float
    host_regret: float
    opponent_regret: float

    @property
    def maximum_regret(self) -> float:
        return max(self.host_regret, self.opponent_regret)


def equilibrium_residual(
    a: np.ndarray, b: np.ndarray, host: np.ndarray, opponent: np.ndarray
) -> EquilibriumResidual:
    a, b, host, opponent = (np.asarray(x, dtype=np.float64) for x in (a, b, host, opponent))
    if a.ndim != 2 or a.shape != b.shape or 0 in a.shape:
        raise ValueError("Expected nonempty aligned payoff matrix pair")
    if host.shape != (a.shape[0],) or opponent.shape != (a.shape[1],):
        raise ValueError("Strategy dimensions do not match role populations")
    if not all(np.isfinite(x).all() for x in (a, b, host, opponent)):
        raise ValueError("Game and probabilities must be finite")
    for strategy in (host, opponent):
        if (strategy < 0).any() or not np.isclose(strategy.sum(), 1, atol=1e-8, rtol=0):
            raise ValueError("Invalid strategy simplex")
    host_actions = a @ opponent
    opponent_actions = host @ b
    host_value = float(host @ host_actions)
    opponent_value = float(opponent_actions @ opponent)
    return EquilibriumResidual(
        host_value,
        opponent_value,
        max(0.0, float(host_actions.max()) - host_value),
        max(0.0, float(opponent_actions.max()) - opponent_value),
    )
