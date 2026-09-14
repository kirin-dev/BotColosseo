"""Summarize completed independent response validation with paired layout CI."""

import argparse
import json
from pathlib import Path

import numpy as np

from botcolosseo.cli.train_hierarchical_strategic import digest
from botcolosseo.evaluation.hierarchical_response import response_gain
from botcolosseo.evaluation.hierarchical_selection import screen_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing validation summary")
    solution = json.loads(args.solution.read_text())
    selection = json.loads((args.validation / "selection.json").read_text())
    results, hashes = {}, {}
    for role in ("host", "opponent"):
        own_hashes = solution["identity"].get(
            f"{role}_strategies", solution["identity"]["strategies"]
        )
        other_role = "opponent" if role == "host" else "host"
        other_hashes = solution["identity"].get(
            f"{other_role}_strategies", solution["identity"]["strategies"]
        )
        own = np.asarray(solution[role])
        target = np.asarray(solution["opponent" if role == "host" else "host"])
        own_ids, target_ids = np.flatnonzero(own), np.flatnonzero(target)
        groups = {}
        seeds = None
        for name in ["candidate", *[f"baseline-{i}" for i in own_ids]]:
            reports = {}
            for index in target_ids:
                path = args.validation / f"{role}-{name}-vs-{index}.json"
                report = json.loads(path.read_text())
                hashes[str(path)] = digest(path)
                expected_first = (
                    selection[role]["sha256"]
                    if name == "candidate"
                    else own_hashes[int(name.split("-")[1])]
                )
                identity = report["identity"]
                if (
                    identity["first"] != expected_first
                    or identity["second"] != other_hashes[index]
                    or identity["executor"] != solution["identity"]["executor"]
                ):
                    raise ValueError("Validation strategy or executor identity mismatch")
                current_seeds = sorted(identity["seeds"])
                seeds = current_seeds if seeds is None else seeds
                if seeds != current_seeds:
                    raise ValueError("Candidate and baseline layouts differ")
                reports[int(index)] = report
            screen_value(reports, role=role, opponent_mixture=target.tolist())
            groups[name] = np.array(
                [
                    [
                        np.mean(
                            [
                                case["first_payoff"]
                                for case in reports[int(index)]["cases"]
                                if case["seed"] == seed and case["first_side"] == role
                            ]
                        )
                        for index in target_ids
                    ]
                    for seed in seeds
                ]
            )
        baseline = np.stack([groups[f"baseline-{i}"] for i in own_ids], axis=1)
        results[role] = response_gain(
            groups["candidate"], baseline, own[own_ids], target[target_ids]
        )
        results[role]["seeds"] = seeds
    output = {
        "results": results,
        "selection": selection,
        "report_hashes": hashes,
        "solution_sha256": digest(args.solution),
        "complete": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2))
    temporary.replace(args.output)
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
