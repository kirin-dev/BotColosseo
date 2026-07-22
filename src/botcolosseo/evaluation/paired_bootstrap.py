from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

FROZEN_BOOTSTRAP_SEED = 20260721
FROZEN_BOOTSTRAP_SAMPLES = 10_000
FROZEN_BOOTSTRAP_CONFIDENCE = 0.95


@dataclass(frozen=True)
class PairedBootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int
    seed: int
    pairs: int


def paired_bootstrap_difference(
    strong: Sequence[float] | np.ndarray,
    baseline: Sequence[float] | np.ndarray,
    *,
    seed: int = FROZEN_BOOTSTRAP_SEED,
    samples: int = FROZEN_BOOTSTRAP_SAMPLES,
    confidence: float = FROZEN_BOOTSTRAP_CONFIDENCE,
) -> PairedBootstrapInterval:
    strong_values = np.asarray(strong, dtype=np.float64)
    baseline_values = np.asarray(baseline, dtype=np.float64)
    if strong_values.ndim != 1 or strong_values.size == 0:
        raise ValueError("Bootstrap requires non-empty paired vectors")
    if baseline_values.shape != strong_values.shape:
        raise ValueError("Bootstrap vectors must have the same shape")
    if not np.isfinite(strong_values).all() or not np.isfinite(baseline_values).all():
        raise ValueError("Bootstrap vectors must be finite")
    if samples <= 0 or seed < 0 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid paired-bootstrap settings")
    differences = strong_values - baseline_values
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, differences.size, size=(samples, differences.size), endpoint=False
    )
    means = differences[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, (tail, 1.0 - tail))
    values = (float(differences.mean()), float(lower), float(upper))
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Bootstrap interval is not finite")
    return PairedBootstrapInterval(
        estimate=values[0],
        lower=values[1],
        upper=values[2],
        confidence=confidence,
        samples=samples,
        seed=seed,
        pairs=int(differences.size),
    )
