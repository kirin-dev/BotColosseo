"""Paired layout-level summaries of deployment-mixture comparisons."""

import numpy as np


def summarize_comparison(report, *, draws=2000, seed=1701):
    if not report.get("complete") or draws < 2:
        raise ValueError("Need a completed comparison and at least two bootstrap draws")
    identity = report["identity"]
    seeds, repeats = identity["seeds"], identity["repeats"]
    methods = ("fixed", "uniform", "meta")
    expected = {
        (s, r, role, m)
        for s in seeds
        for r in range(repeats)
        for role in ("host", "opponent")
        for m in methods
    }
    cases = {(r["seed"], r["repeat"], r["role"], r["method"]): r for r in report["cases"]}
    if len(seeds) < 2 or len({s % 128 for s in seeds}) != len(seeds):
        raise ValueError("Need distinct layout clusters")
    if len(cases) != len(report["cases"]) or cases.keys() != expected or repeats <= 0:
        raise ValueError("Incomplete or duplicate paired case grid")
    for s in seeds:
        for r in range(repeats):
            for role in ("host", "opponent"):
                group = [cases[s, r, role, m] for m in methods]
                if len({c["opponent_index"] for c in group}) != 1:
                    raise ValueError("Methods did not face the same opponent draw")
                for c in group:
                    value = c["payoff"]
                    if (
                        c["layout"] != s % 128
                        or not np.isfinite(value)
                        or not 0 <= value <= 1
                        or value != c["payoffs"][int(role == "opponent")]
                    ):
                        raise ValueError("Invalid role payoff or layout")
    summary, clusters = {}, {}
    for method in methods:
        values = np.array(
            [
                cases[s, r, role, method]["payoff"]
                for s in seeds
                for r in range(repeats)
                for role in ("host", "opponent")
            ]
        )
        clusters[method] = values.reshape(len(seeds), -1).mean(1)
        summary[method] = {
            "games": len(values),
            "mean_payoff": float(values.mean()),
            "mean_banked": float(values.mean() * 150),
            "positive_banked_rate": float((values > 0).mean()),
            "role_mean_payoff": {
                role: float(
                    np.mean(
                        [cases[s, r, role, method]["payoff"] for s in seeds for r in range(repeats)]
                    )
                )
                for role in ("host", "opponent")
            },
        }
    samples = np.random.default_rng(seed).integers(len(seeds), size=(draws, len(seeds)))
    differences = {}
    for first, second in (("uniform", "fixed"), ("meta", "fixed"), ("meta", "uniform")):
        delta = clusters[first] - clusters[second]
        differences[f"{first}_minus_{second}"] = {
            "mean_payoff_delta": float(delta.mean()),
            "layout_bootstrap_95_interval": np.quantile(
                delta[samples].mean(1), [0.025, 0.975]
            ).tolist(),
        }
    return {
        "methods": summary,
        "paired_differences": differences,
        "layout_clusters": len(seeds),
        "draws": draws,
        "bootstrap_seed": seed,
        "scope": "Neutral/Hard vs common uniform opponents; not global exploitability",
    }
