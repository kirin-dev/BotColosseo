"""Paired case-cluster uncertainty for restricted-game mixture estimates."""

from __future__ import annotations

import numpy as np

from botcolosseo.training.hierarchical_game import solve_bimatrix
from botcolosseo.training.hierarchical_population import (
    PayoffCase,
    PopulationWindow,
    payoff_matrices,
)
from botcolosseo.training.hierarchical_protocol import own_payoffs


def bootstrap_meta_strategy(
    window: PopulationWindow,
    cases: list[PayoffCase],
    *,
    required_cases: frozenset[tuple[int, int, int]],
    draws: int = 200,
    seed: int = 1701,
) -> dict:
    """Resample paired seed/layout clusters across every cell; keep repeats together.

    Quantiles summarize mixture sensitivity, not a confidence region for a unique
    true equilibrium (games may admit several equilibria).
    """
    if draws < 2:
        raise ValueError("Need at least two bootstrap draws")
    a, b = payoff_matrices(window, cases, required_cases=required_cases)
    point = solve_bimatrix(a, b)
    clusters = sorted({(case_seed, layout) for case_seed, layout, _ in required_cases})
    if len(clusters) < 2:
        raise ValueError("Need at least two seed/layout clusters")
    cell_cluster = {
        (h, o, k): []
        for h in window.host_policies
        for o in window.opponent_policies
        for k in clusters
    }
    for case in cases:
        cell_cluster[(case.host_policy, case.opponent_policy, (case.seed, case.layout))].append(
            own_payoffs(
                case.host_banked, case.opponent_banked, globally_settled=case.globally_settled
            )
        )
    cube = np.zeros((len(clusters), *a.shape, 2))
    for k, cluster in enumerate(clusters):
        for i, h in enumerate(window.host_policies):
            for j, o in enumerate(window.opponent_policies):
                cube[k, i, j] = np.mean(cell_cluster[(h, o, cluster)], axis=0)
    # Equal cluster weights require equal repeat counts for the point estimate.
    sizes = [
        sum((case_seed, layout) == cluster for case_seed, layout, _ in required_cases)
        for cluster in clusters
    ]
    if len(set(sizes)) != 1:
        raise ValueError("Unequal repeat counts require an explicit cluster weighting protocol")
    rng = np.random.default_rng(seed)
    host, opponent, residuals, methods, entropy_status = [], [], [], [], []
    for _ in range(draws):
        sampled = cube[rng.integers(0, len(clusters), len(clusters))].mean(axis=0)
        result = solve_bimatrix(sampled[:, :, 0], sampled[:, :, 1])
        host.append(result.host)
        opponent.append(result.opponent)
        residuals.append(result.residual.maximum_regret)
        methods.append(result.method)
        entropy_status.append(result.entropy_refinement_complete)
    return {
        "clusters": len(clusters),
        "draws": draws,
        "seed": seed,
        "host_point": point.host.tolist(),
        "opponent_point": point.opponent.tolist(),
        "host_weight_quantiles": np.quantile(host, [0.025, 0.5, 0.975], axis=0).tolist(),
        "opponent_weight_quantiles": np.quantile(opponent, [0.025, 0.5, 0.975], axis=0).tolist(),
        "maximum_sample_regret": max(residuals),
        "approximate_draws": sum(m.startswith("approximate") for m in methods),
        "entropy_incomplete_draws": sum(not s for s in entropy_status),
        "interpretation": "paired seed/layout bootstrap mixture sensitivity",
    }
