from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping


def pfsp_probabilities(
    win_rates: Mapping[str, float], *, floor: float = 0.05
) -> dict[str, float]:
    if not win_rates:
        raise ValueError("PFSP requires at least one payoff")
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("PFSP floor must be positive")
    weights: dict[str, float] = {}
    for policy_id in sorted(win_rates):
        win_rate = win_rates[policy_id]
        if not policy_id or not math.isfinite(win_rate) or not 0.0 <= win_rate <= 1.0:
            raise ValueError("PFSP win rates must be finite probabilities")
        weights[policy_id] = max(floor, (1.0 - win_rate) ** 2)
    total = sum(weights.values())
    return {policy_id: weight / total for policy_id, weight in weights.items()}


def stable_uniform(
    *,
    master_seed: int,
    pair_slot: int,
    pool_hash: str,
    payoff_hash: str,
    stream: str,
) -> float:
    if pair_slot < 0 or not pool_hash or not payoff_hash or not stream:
        raise ValueError("Invalid deterministic sampler identity")
    payload = json.dumps(
        {
            "master_seed": master_seed,
            "pair_slot": pair_slot,
            "pool_hash": pool_hash,
            "payoff_hash": payoff_hash,
            "stream": stream,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / 2**64
