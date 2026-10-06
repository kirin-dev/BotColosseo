"""Matched-prefix closed-loop style diagnostics for a frozen deployment."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from botcolosseo.demo.control_trace_audit import audit_control_report


def summarize(directory):
    paths = {name: directory / f"{name}.json"
             for name in ("neutral", "aggressive", "defensive", "explorer")}
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    baseline = reports["neutral"]
    identity = {key: baseline[key] for key in ("executor", "strategy", "opponent")}
    cases = {}
    for name, report in reports.items():
        if not report["complete"] or any(report[key] != value for key, value in identity.items()):
            raise ValueError("Incomplete report or different deployment")
        indexed = {(c["seed"], c["role"]): c for c in report["cases"]}
        if len(indexed) != len(report["cases"]):
            raise ValueError("Duplicate case")
        cases[name] = indexed
        if name != "neutral":
            audit_control_report(report)
            if indexed.keys() != cases["neutral"].keys():
                raise ValueError("Unpaired cases")
    result = {"identities": identity, "source_sha256": {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "scope": "Seed/role-paired development cases; prefix audit limits causal interpretation",
        "window": [88, 161], "styles": {}}
    for name in ("aggressive", "defensive", "explorer"):
        rows = []
        for key, switched in cases[name].items():
            original = cases["neutral"][key]
            # Compare both players' observations entering the intervention decision.
            prefix = original["observation_hashes"][:82] == switched["observation_hashes"][:82]
            prefix &= [(r["action"], r["command"]) for r in original["control_trace"][:81]] == [
                (r["action"], r["command"]) for r in switched["control_trace"][:81]]
            row = {"seed": key[0], "role": key[1], "bank_control": original["payoff"] * 150,
                   "bank_switch": switched["payoff"] * 150, "prefix_equal": bool(prefix)}
            row["first_observation_difference"] = next((i for i, (left, right) in enumerate(zip(
                original["observation_hashes"][:82], switched["observation_hashes"][:82],
                strict=False)) if left != right), None)
            for arm, case in (("control", original), ("switch", switched)):
                window = case["control_trace"][88:161]
                row[arm] = {"observed_steps": len(window), **{
                    metric: float(np.mean([predicate(r) for r in window])) if window else None
                    for metric, predicate in {
                        "attack": lambda r: r["action"] >= 9,
                        "search": lambda r: r["command"] < 3,
                        "extract": lambda r: r["command"] >= 5,
                    }.items()}}
            rows.append(row)
        seeds = sorted({r["seed"] for r in rows})
        delta = np.array([np.mean([r["bank_switch"] - r["bank_control"]
                                  for r in rows if r["seed"] == seed]) for seed in seeds])
        rng = np.random.default_rng(1701)
        samples = rng.choice(delta, (2000, len(delta)), replace=True).mean(axis=1)
        result["styles"][name] = {
            "cases": rows, "mean_bank_delta": float(delta.mean()),
            "matched_prefix_cases": sum(r["prefix_equal"] for r in rows),
            "causal_matched_history_comparison": all(r["prefix_equal"] for r in rows),
            "layout_bootstrap_95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "control_audit": reports[name]["control_audit"],
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
