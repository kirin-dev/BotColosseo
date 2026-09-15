"""Reproducible exogenous condition segments, independent of game state."""

import numpy as np

from botcolosseo.training.hierarchical_protocol import ControlCondition

STYLES = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
          (1, 1, 0), (1, 0, 1), (0, 1, 1))
LEVELS = (0.0, 0.5, 1.0)


def random_control_schedule(*, seed, mode, initial, horizon=657,
                            min_segment=40, max_segment=100):
    """Build requested controls; controllers apply them at their own boundaries.

    Style-only preserves initial difficulty; difficulty-only preserves style.
    Joint changes both inputs. No model, environment state, or payoff is read.
    """
    if mode not in ("style", "difficulty", "joint"):
        raise ValueError("Unknown switch curriculum mode")
    if not isinstance(initial, ControlCondition):
        raise ValueError("Initial condition must be validated")
    if any(type(x) is not int for x in (seed, horizon, min_segment, max_segment)):
        raise ValueError("Seed and decision counts must be integers")
    if seed < 0 or not 0 < min_segment <= max_segment < horizon:
        raise ValueError("Invalid random segment bounds")
    rng = np.random.default_rng(seed)
    schedule = [(0, initial)]
    start = 0
    current = initial
    while True:
        start += int(rng.integers(min_segment, max_segment + 1))
        if start >= horizon:
            return schedule
        style = current.as_tuple()[:3]
        difficulty = current.difficulty
        if mode in ("style", "joint"):
            choices = [value for value in STYLES if value != style]
            style = choices[int(rng.integers(len(choices)))]
        if mode in ("difficulty", "joint"):
            choices = [value for value in LEVELS if value != difficulty]
            difficulty = choices[int(rng.integers(len(choices)))]
        current = ControlCondition(*style, difficulty)
        schedule.append((start, current))
