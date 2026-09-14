"""Run all ordered strategic pairs on bounded workers; retain role-specific cells."""

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from botcolosseo.cli.train_hierarchical_strategic import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--strategies", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(96, 108)))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--host-indices", type=int, nargs="+")
    parser.add_argument("--opponent-indices", type=int, nargs="+")
    parser.add_argument("--reuse-directory", type=Path)
    args = parser.parse_args()
    if len(args.strategies) < 2 or len(set(args.strategies)) != len(args.strategies):
        raise ValueError("Need distinct strategy paths")
    rows = args.host_indices or list(range(len(args.strategies)))
    columns = args.opponent_indices or list(range(len(args.strategies)))
    for indices in (rows, columns):
        if (
            len(set(indices)) != len(indices)
            or len(indices) > 6
            or any(index < 0 or index >= len(args.strategies) for index in indices)
        ):
            raise ValueError("Invalid role population indices")
    identity = {
        "executor": digest(args.executor),
        "strategies": [digest(path) for path in args.strategies],
        "seeds": args.seeds,
        "repeats": args.repeats,
        "runner_source": digest(Path(__file__)),
        "pair_source": digest(Path(__file__).with_name("evaluate_hierarchical_pair.py")),
    }
    identity["host_strategies"] = [identity["strategies"][i] for i in rows]
    identity["opponent_strategies"] = [identity["strategies"][i] for i in columns]
    args.output.mkdir(parents=True, exist_ok=True)
    identity_path = args.output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Matrix directory identity mismatch")
    identity_path.write_text(json.dumps(identity, indent=2))
    needed = {(i, j) for i in rows for j in columns}
    pairs = sorted(needed | {(j, i) for i, j in needed})

    def worker(worker_index):
        for i, j in pairs[worker_index :: len(args.devices)]:
            output = args.output / f"pair-{i}-{j}.json"
            if args.reuse_directory and not output.exists():
                previous = args.reuse_directory / output.name
                if previous.exists():
                    shutil.copy2(
                        previous, output
                    )  # Pair runner verifies full identity before reuse.
            command = [
                sys.executable,
                "-u",
                "-m",
                "botcolosseo.cli.evaluate_hierarchical_pair",
                "--executor",
                str(args.executor),
                "--first",
                str(args.strategies[i]),
                "--second",
                str(args.strategies[j]),
                "--output",
                str(output),
                "--device",
                args.devices[worker_index],
                "--repeats",
                str(args.repeats),
                "--seeds",
                *map(str, args.seeds),
            ]
            if output.exists():
                command.append("--resume")
            with (args.output / f"pair-{i}-{j}.log").open("a") as log:
                completed = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            print(json.dumps({"pair": [i, j], "exit_code": completed.returncode}), flush=True)
            if completed.returncode:
                raise RuntimeError(f"Pair {i},{j} failed; completed artifacts preserved")

    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        list(pool.map(worker, range(len(args.devices))))
    n = len(args.strategies)
    cells = [[[] for _ in range(n)] for _ in range(n)]
    for i, j in pairs:
        report = json.loads((args.output / f"pair-{i}-{j}.json").read_text())
        expected = len(args.seeds) * args.repeats * 2
        if not report["complete"] or len(report["cases"]) != expected:
            raise ValueError("Incomplete pair cannot enter matrix")
        for case in report["cases"]:
            host, opponent = (i, j) if case["first_side"] == "host" else (j, i)
            cells[host][opponent].append([case["host_payoff"], case["opponent_payoff"]])
    result = {
        "identity": identity,
        "complete": True,
        "row_role": "host",
        "column_role": "opponent",
        "counts": [[len(cells[i][j]) for j in columns] for i in rows],
    }
    for role, key in enumerate(("A", "B")):
        result[key] = [
            [sum(value[role] for value in cells[i][j]) / len(cells[i][j]) for j in columns]
            for i in rows
        ]
    temporary = args.output / "matrix.tmp"
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(args.output / "matrix.json")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
