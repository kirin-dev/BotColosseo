"""Condition-indexed opponent targets for one shared strategic Actor."""

import math

import numpy as np

from botcolosseo.training.hierarchical_protocol import matrix_condition


def conditional_marginals(solutions, conditions, *, executor, population, learner_role):
    """Return targets in training-condition order; never broadcast Neutral sigma."""
    if learner_role not in ("host", "opponent"):
        raise ValueError("Invalid learner role")
    other = "opponent" if learner_role == "host" else "host"
    targets = {}
    for solution in solutions:
        identity = solution["identity"]
        condition = matrix_condition(identity)
        if condition in targets:
            raise ValueError("Duplicate condition solution")
        if identity["executor"] != executor:
            raise ValueError("Conditional solution executor mismatch")
        if identity.get(f"{other}_strategies", identity["strategies"]) != population:
            raise ValueError("Conditional opponent population ordering mismatch")
        regret = solution["maximum_regret"]
        if not math.isfinite(regret) or not 0 <= regret <= 1e-3:
            raise ValueError("Unacceptable conditional game residual")
        marginal = np.asarray(solution[other], dtype=float)
        if (
            marginal.shape != (len(population),)
            or not np.isfinite(marginal).all()
            or (marginal < 0).any()
            or not np.isclose(marginal.sum(), 1, atol=1e-8, rtol=0)
        ):
            raise ValueError("Invalid conditional opponent marginal")
        targets[condition] = marginal
    if set(targets) != set(conditions) or len(conditions) != len(set(conditions)):
        raise ValueError("Every training condition needs its own solved game")
    return np.stack([targets[condition] for condition in conditions])
