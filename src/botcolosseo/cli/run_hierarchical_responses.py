"""After a complete matrix, solve and train one response for each role."""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from botcolosseo.cli.solve_hierarchical_matrix import solve_report
from botcolosseo.cli.train_hierarchical_strategic import digest
from botcolosseo.training.hierarchical_protocol import require_neutral_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--population", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--devices", nargs=2, default=["cuda:0", "cuda:1"])
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("Positive response budget required")
    started = time.monotonic()
    while not args.matrix.exists():
        if args.wait_pid is None or time.monotonic() - started > 7200:
            raise RuntimeError("Completed matrix unavailable; no response started")
        os.kill(args.wait_pid, 0)  # Missing producer is a failure, never fabricate a matrix.
        time.sleep(30)
    matrix = json.loads(args.matrix.read_text())
    require_neutral_matrix(matrix["identity"])
    if matrix["identity"]["executor"] != digest(args.executor):
        raise ValueError("Matrix executor mismatch")
    if matrix["identity"]["strategies"] != [digest(path) for path in args.population]:
        raise ValueError("Matrix population ordering mismatch")
    solution = solve_report(matrix)
    solution.update({"matrix_sha256": digest(args.matrix), "identity": matrix["identity"]})
    args.output.mkdir(parents=True, exist_ok=True)
    solution_path = args.output / "solution.json"
    if solution_path.exists():
        previous = json.loads(solution_path.read_text())
        if previous != solution:
            raise ValueError("Existing response solution differs")
    else:
        temporary = solution_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(solution, indent=2))
        temporary.replace(solution_path)
    # Warm-start each role from its strongest existing pure strategy against the
    # fixed target marginal. This selection uses the initial matrix, not validation.
    initial_indices = [
        int(np.argmax(np.asarray(matrix["A"]) @ np.asarray(solution["opponent"]))),
        int(np.argmax(np.asarray(solution["host"]) @ np.asarray(matrix["B"]))),
    ]
    by_hash = {digest(path): path for path in args.population}
    role_populations = {
        role: [
            by_hash[key]
            for key in matrix["identity"].get(
                f"{role}_strategies", matrix["identity"]["strategies"]
            )
        ]
        for role in ("host", "opponent")
    }

    def train(index):
        role = ("host", "opponent")[index]
        output = args.output / role
        command = [
            sys.executable,
            "-u",
            "-m",
            "botcolosseo.cli.train_hierarchical_strategic",
            "--executor",
            str(args.executor),
            "--solution",
            str(solution_path),
            "--initial",
            str(role_populations[role][initial_indices[index]]),
            "--output",
            str(output),
            "--steps",
            str(args.steps),
            "--role",
            role,
            "--device",
            args.devices[index],
            "--population",
            *map(str, role_populations["opponent" if role == "host" else "host"]),
        ]
        if output.exists():
            command.append("--resume")
        print(
            json.dumps(
                {"role": role, "initial_index": initial_indices[index], "budget": args.steps}
            ),
            flush=True,
        )
        with (args.output / f"{role}.log").open("a") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        print(json.dumps({"role": role, "exit_code": result.returncode}), flush=True)
        if result.returncode:
            raise RuntimeError(f"{role} response failed; checkpoints preserved")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(train, range(2)))


if __name__ == "__main__":
    main()
