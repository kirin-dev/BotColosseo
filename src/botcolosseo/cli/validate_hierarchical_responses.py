"""Freeze screen-selected responses, then collect independent validation games."""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from botcolosseo.cli.train_hierarchical_strategic import digest
from botcolosseo.evaluation.hierarchical_role_paths import role_paths
from botcolosseo.evaluation.hierarchical_selection import screen_value
from botcolosseo.training.hierarchical_protocol import require_neutral_matrix


def main():
    parser = argparse.ArgumentParser()
    for name in ("executor", "responses", "screen", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--population", type=Path, nargs="+", required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(112, 120)))
    args = parser.parse_args()
    solution = json.loads((args.responses / "solution.json").read_text())
    require_neutral_matrix(solution["identity"])
    populations = role_paths(solution["identity"], args.population)
    if solution["identity"]["strategies"] != [digest(path) for path in args.population]:
        raise ValueError("Population differs from solved game")
    candidates = {
        role: sorted((args.responses / role).glob("candidate-*.pt"))
        for role in ("host", "opponent")
    }
    required = [
        args.screen / f"{role}-{path.stem}-vs-{index}.json"
        for role, paths in candidates.items()
        for path in paths
        for index, probability in enumerate(solution["opponent" if role == "host" else "host"])
        if probability > 0
    ]
    if any(not paths for paths in candidates.values()):
        raise ValueError("Both roles require trained candidates")
    started = time.monotonic()
    while not all(path.exists() and json.loads(path.read_text())["complete"] for path in required):
        if args.wait_pid is None or time.monotonic() - started > 7200:
            raise RuntimeError("Screen incomplete; no candidate selected")
        os.kill(args.wait_pid, 0)
        time.sleep(30)
    selected, selection = {}, {}
    validation_seeds = args.seeds
    if len(set(validation_seeds)) != len(validation_seeds) or len(validation_seeds) < 2:
        raise ValueError("Independent validation requires distinct layout seeds")
    for role, paths in candidates.items():
        target = solution["opponent" if role == "host" else "host"]
        scores = []
        for path in paths:
            reports = {
                i: json.loads((args.screen / f"{role}-{path.stem}-vs-{i}.json").read_text())
                for i, probability in enumerate(target)
                if probability > 0
            }
            for report in reports.values():
                if report["identity"]["first"] != digest(path):
                    raise ValueError("Screen candidate hash mismatch")
                if set(report["identity"]["seeds"]) & set(validation_seeds):
                    raise ValueError("Independent validation overlaps checkpoint selection")
            scores.append(
                (screen_value(reports, role=role, opponent_mixture=target)["mean_payoff"], path)
            )
        value, path = max(scores, key=lambda item: item[0])  # Earlier checkpoint wins an exact tie.
        selected[role] = path
        selection[role] = {
            "checkpoint": str(path),
            "sha256": digest(path),
            "screen_value": value,
            "all_scores": {str(path): value for value, path in scores},
        }
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = args.output / "selection.json"
    if frozen.exists() and json.loads(frozen.read_text()) != selection:
        raise ValueError("Preserving already frozen selection")
    frozen.write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection), flush=True)

    def worker(role_index):
        role = ("host", "opponent")[role_index]
        target = solution["opponent" if role == "host" else "host"]
        policies = [("candidate", selected[role])]
        policies += [
            (f"baseline-{i}", populations[role][i]) for i, p in enumerate(solution[role]) if p > 0
        ]
        for name, path in policies:
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
                    str(path),
                    "--second",
                    str(populations["opponent" if role == "host" else "host"][index]),
                    "--output",
                    str(output),
                    "--device",
                    f"cuda:{role_index}",
                    "--repeats",
                    "2",
                    "--seeds",
                    *map(str, validation_seeds),
                ]
                if output.exists():
                    command.append("--resume")
                with output.with_suffix(".log").open("a") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                print(
                    json.dumps({"role": role, "policy": name, "opponent": index, "complete": True}),
                    flush=True,
                )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, range(2)))


if __name__ == "__main__":
    main()
