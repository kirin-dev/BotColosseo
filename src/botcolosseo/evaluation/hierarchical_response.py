"""Paired layout-cluster response gains under one fixed target mixture."""

import numpy as np


def response_gain(candidate, baseline, own_mixture, opponent_mixture, *, draws=2000, seed=1701):
    """Inputs are per-layout means after averaging repeated games within layout.

    candidate: [layout, opponent]; baseline: [layout, own_strategy, opponent].
    Caller orients both arrays to the learner role and uses independent validation
    layouts, never the matrix's training/selection layouts. Mixtures stay fixed
    during bootstrap: this interval is response gain, not equilibrium uncertainty.
    """
    candidate, baseline, own, opponent = (
        np.asarray(value, dtype=float)
        for value in (candidate, baseline, own_mixture, opponent_mixture)
    )
    if candidate.ndim != 2 or baseline.ndim != 3 or candidate.shape[0] < 2:
        raise ValueError("Need at least two paired layout clusters")
    if baseline.shape != (candidate.shape[0], own.size, opponent.size):
        raise ValueError("Baseline axes disagree with mixtures/layouts")
    if candidate.shape[1] != opponent.size or draws < 100:
        raise ValueError("Invalid opponent dimension or bootstrap budget")
    for probabilities in (own, opponent):
        if (probabilities.ndim != 1 or not np.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or not np.isclose(probabilities.sum(), 1, atol=1e-8, rtol=0)):
            raise ValueError("Invalid target mixture")
    for payoffs in (candidate, baseline):
        if not np.isfinite(payoffs).all() or (payoffs < 0).any() or (payoffs > 1).any():
            raise ValueError("Missing or invalid own-bank payoff")
    candidate_values = candidate @ opponent
    baseline_values = np.einsum("i,lij,j->l", own, baseline, opponent)
    differences = candidate_values - baseline_values
    indices = np.random.default_rng(seed).integers(0, len(differences), (draws, len(differences)))
    interval = np.quantile(differences[indices].mean(axis=1), [0.025, 0.975])
    gain = float(differences.mean())
    return {
        "layout_clusters": len(differences), "candidate_value": float(candidate_values.mean()),
        "baseline_value": float(baseline_values.mean()), "gain": gain,
        "gain_ci95": interval.tolist(), "practical_threshold": 0.02,
        "improvement_supported": bool(gain >= 0.02 and interval[0] > 0),
        "layout_gains": differences.tolist(), "bootstrap_draws": draws,
        "scope": "paired validation gain against fixed mixture; not global exploitability",
    }
