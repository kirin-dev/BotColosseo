"""Validation-only checkpoint screen against the solved opponent support."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from botcolosseo.evaluation.hierarchical_role_paths import role_paths
from botcolosseo.training.hierarchical_protocol import require_neutral_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--population", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[108, 109, 110, 111])
    args = parser.parse_args()
    solution = json.loads((args.responses / "solution.json").read_text())
    require_neutral_matrix(solution["identity"])
    populations = role_paths(solution["identity"], args.population)
    if set(args.seeds) & set(solution["identity"]["seeds"]):
        raise ValueError("Screen layouts must be independent of initial matrix")
    args.output.mkdir(parents=True, exist_ok=True)

    def worker(role_index):
        role = ("host", "opponent")[role_index]
        target = solution["opponent" if role == "host" else "host"]
        own = solution[role]
        # Include every supported original strategy as baseline; no assume-pure shortcut.
        candidates = [
            (f"baseline-{i}", path) for i, path in enumerate(populations[role]) if own[i] > 0
        ]
        candidates += [
            (path.stem, path) for path in sorted((args.responses / role).glob("candidate-*.pt"))
        ]
        for name, candidate in candidates:
            for index, probability in enumerate(target):
                if probability == 0:
                    continue
                output = args.output / f"{role}-{name}-vs-{index}.json"
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "botcolosseo.cli.evaluate_hierarchical_pair",
                    "--executor",
                    str(args.executor),
                    "--first",
                    str(candidate),
                    "--second",
                    str(populations["opponent" if role == "host" else "host"][index]),
                    "--output",
                    str(output),
                    "--device",
                    f"cuda:{role_index}",
                    "--repeats",
                    "1",
                    "--seeds",
                    *map(str, args.seeds),
                ]
                if output.exists():
                    command.append("--resume")
                with output.with_suffix(".log").open("a") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                report = json.loads(output.read_text())
                values = [
                    case["first_payoff"] for case in report["cases"] if case["first_side"] == role
                ]
                print(
                    json.dumps(
                        {
                            "role": role,
                            "candidate": name,
                            "opponent": index,
                            "cases": len(values),
                            "value": float(np.mean(values)),
                        }
                    ),
                    flush=True,
                )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, range(2)))


if __name__ == "__main__":
    main()
