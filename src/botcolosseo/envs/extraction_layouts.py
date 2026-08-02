from __future__ import annotations

import math

RANDOMIZED_LAYOUT_COUNT = 128
RANDOMIZED_LOOT_VALUES = (10, 10, 10, 10, 25, 25, 50)
RANDOMIZED_LOOT_ANCHORS = (
    (-520.0, -288.0),
    (-520.0, 0.0),
    (-520.0, 288.0),
    (-384.0, -224.0),
    (-384.0, 96.0),
    (-192.0, -224.0),
    (-192.0, 96.0),
    (0.0, -224.0),
    (0.0, 96.0),
    (192.0, -224.0),
    (192.0, 96.0),
    (384.0, -224.0),
    (384.0, 96.0),
    (520.0, -288.0),
    (520.0, 0.0),
    (520.0, 288.0),
)
_COPRIME_MULTIPLIERS = (1, 3, 5, 7, 9, 11, 13, 15)


def randomized_layout_variant(seed: int) -> int:
    if seed < 0:
        raise ValueError("Extraction layout seed must be nonnegative")
    return seed % RANDOMIZED_LAYOUT_COUNT


def randomized_loot_layout(
    variant: int,
) -> tuple[tuple[int, float, float], ...]:
    if not 0 <= variant < RANDOMIZED_LAYOUT_COUNT:
        raise ValueError("Randomized Extraction layout variant is out of range")
    multiplier = _COPRIME_MULTIPLIERS[variant % len(_COPRIME_MULTIPLIERS)]
    offset = variant // len(_COPRIME_MULTIPLIERS)
    return tuple(
        (value, *RANDOMIZED_LOOT_ANCHORS[(multiplier * index + offset) % 16])
        for index, value in enumerate(RANDOMIZED_LOOT_VALUES)
    )


def strong_randomized_route(
    *, side: str, variant: int
) -> tuple[tuple[float, float], ...]:
    if side not in {"host", "opponent"}:
        raise ValueError(f"Unsupported extraction side: {side}")
    layout = randomized_loot_layout(variant)
    remaining = [layout[index][1:] for index in (6, 4, 0)]
    current = (-640.0, 0.0) if side == "host" else (640.0, 0.0)
    route: list[tuple[float, float]] = []
    while remaining:
        target = min(remaining, key=lambda point: math.dist(current, point))
        remaining.remove(target)
        route.append(target)
        current = target
    return tuple(route)
