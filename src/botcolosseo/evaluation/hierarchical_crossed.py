"""Audit crossed reset demonstrations without assuming seed-level determinism."""

import math


def audit_crossed_search(report: dict, *, seeds: frozenset[int]) -> dict:
    if not report.get("complete") or report.get("privileged_oracle") is not True:
        raise ValueError("Need a completed explicitly privileged demonstration report")
    if not seeds:
        raise ValueError("Expected seed set must be nonempty")
    commands = ("SEARCH_NORTH", "SEARCH_CENTER", "SEARCH_SOUTH")
    expected = {(s, role, c) for s in seeds for role in ("host", "opponent") for c in commands}
    rows = report["cases"]
    indexed = {(r["seed"], r["side"], r["command"]): r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("Crossed case grid is missing, duplicated or unexpected")
    equal = 0
    mismatches = []
    for seed in sorted(seeds):
        for side in ("host", "opponent"):
            group = [indexed[(seed, side, c)] for c in commands]
            for row in group:
                digest = row["initial_frame_sha256"]
                scalars = row["initial_scalars"]
                if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("Invalid initial-frame digest")
                if len(scalars) != 9 or not all(math.isfinite(v) for v in scalars):
                    raise ValueError("Invalid initial own-state vector")
            signatures = {(r["initial_frame_sha256"], tuple(r["initial_scalars"])) for r in group}
            if len(signatures) == 1:
                equal += 1
            else:
                mismatches.append({"seed": seed, "side": side})
    return {
        "cases": len(rows),
        "groups": 2 * len(seeds),
        "equal_initial_fair_observation_groups": equal,
        "mismatched_groups": mismatches,
        "applicable_by_command": {
            c: sum(r["applicable"] for r in rows if r["command"] == c) for c in commands
        },
        "scope": "initial fair-observation equality only; not hidden-world or trajectory equality",
    }
