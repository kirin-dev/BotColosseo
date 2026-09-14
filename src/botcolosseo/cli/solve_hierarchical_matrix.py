"""Bind restricted bimatrix solutions to completed empirical matrix artifacts."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from botcolosseo.cli.train_hierarchical_strategic import digest
from botcolosseo.training.hierarchical_game import solve_bimatrix


def solve_report(report):
    if report.get("complete") is not True:
        raise ValueError("Cannot solve incomplete cross-play")
    if report.get("row_role") != "host" or report.get("column_role") != "opponent":
        raise ValueError("Explicit canonical role axes required")
    a, b, counts = (np.asarray(report[key], dtype=float) for key in ("A", "B", "counts"))
    identity = report["identity"]
    shape = tuple(
        len(identity.get(f"{role}_strategies", identity["strategies"]))
        for role in ("host", "opponent")
    )
    expected = len(report["identity"]["seeds"]) * report["identity"]["repeats"] * 2
    if expected <= 0 or any(value.shape != shape for value in (a, b, counts)):
        raise ValueError("Invalid population or matrix shape")
    if not np.all(counts == expected):
        raise ValueError("Every canonical role cell needs the complete case budget")
    if any(
        not np.isfinite(value).all() or (value < 0).any() or (value > 1).any() for value in (a, b)
    ):
        raise ValueError("Own-bank payoff outside [0,1]")
    solution = solve_bimatrix(a, b)
    return {
        "host": solution.host.tolist(),
        "opponent": solution.opponent.tolist(),
        "method": solution.method,
        "residual": asdict(solution.residual),
        "maximum_regret": solution.residual.maximum_regret,
        "entropy_refinement_complete": solution.entropy_refinement_complete,
        "scope": "restricted estimated game only; not full-game Nash convergence",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing solution")
    matrix = json.loads(args.matrix.read_text())
    result = solve_report(matrix)
    result["matrix_sha256"] = digest(args.matrix)
    result["identity"] = matrix["identity"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(args.output)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
