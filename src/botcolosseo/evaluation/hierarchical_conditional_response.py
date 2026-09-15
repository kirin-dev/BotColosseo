"""Paired layout-cluster summaries for sampled condition-specific response tests."""

import numpy as np


def summarize(report, *, draws=2000):
    if not report.get("complete") or draws < 100:
        raise ValueError("Complete response evaluation and adequate bootstrap draws required")
    identity = report["identity"]
    conditions, seeds = identity["conditions"], identity["seeds"]
    repeats = identity["repeats"]
    if not conditions or not seeds or repeats <= 0:
        raise ValueError("Empty evaluation grid")
    if identity["role"] not in ("host", "opponent") or len({s % 128 for s in seeds}) != len(seeds):
        raise ValueError("Invalid role or repeated layout clusters")
    expected = {(c, s, r, m) for c in range(len(conditions)) for s in seeds
                for r in range(repeats) for m in ("baseline", "candidate")}
    rows = {(r["condition_index"], r["seed"], r["repeat"], r["method"]): r
            for r in report["cases"]}
    if len(rows) != len(report["cases"]) or set(rows) != expected:
        raise ValueError("Incomplete or duplicate paired grid")
    values = np.empty((len(conditions), len(seeds), repeats, 2))
    for c in range(len(conditions)):
        for s_index, seed in enumerate(seeds):
            for repeat in range(repeats):
                pair = [rows[c, seed, repeat, m] for m in ("baseline", "candidate")]
                if any(pair[0][k] != pair[1][k] for k in ("opponent_index", "baseline_index")):
                    raise ValueError("Paired methods must share sampled identities")
                for method, row in enumerate(pair):
                    payoff = row["payoff"]
                    if (row["condition"] != conditions[c] or row["layout"] != seed % 128
                            or not np.isfinite(payoff) or not 0 <= payoff <= 1
                            or payoff != row["payoffs"][int(identity["role"] == "opponent")]):
                        raise ValueError("Invalid condition, layout or learner payoff")
                    values[c, s_index, repeat, method] = payoff
    # Average repeats first. The overall score gives each registered condition equal weight.
    layout_values = values.mean(axis=2)
    indices = np.random.default_rng(1701).integers(0, len(seeds), (draws, len(seeds)))

    def metrics(pair_values):
        delta = pair_values[:, 1] - pair_values[:, 0]
        interval = (
            np.quantile(delta[indices].mean(axis=1), [0.025, 0.975]).tolist()
            if len(seeds) >= 2 else None
        )
        return {
            "baseline_payoff": float(pair_values[:, 0].mean()),
            "candidate_payoff": float(pair_values[:, 1].mean()),
            "gain": float(delta.mean()), "gain_ci95": interval,
            "layout_gains": delta.tolist(),
            "improvement_supported": bool(
                interval is not None and delta.mean() >= 0.02 and interval[0] > 0
            ),
        }

    return {
        "role": identity["role"], "layout_clusters": len(seeds),
        "scope": "paired sampled-mixture response gain; not global exploitability",
        "overall": metrics(layout_values.mean(axis=0)),
        "by_condition": [
            {"condition": condition, **metrics(layout_values[c])}
            for c, condition in enumerate(conditions)
        ],
        "bootstrap_draws": draws, "practical_threshold": 0.02,
    }
