"""Compare old and candidate executors under identical fixed command protocols."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from botcolosseo.cli.train_hierarchical_strategic import digest
from botcolosseo.evaluation.hierarchical_executor import summarize_executor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(80, 88)))
    args = parser.parse_args()
    checkpoints = [args.baseline, *sorted(args.run.glob("candidate-*.pt"))]
    if len(checkpoints) < 2:
        raise ValueError("Need completed candidate checkpoints")
    args.output.mkdir(parents=True, exist_ok=True)

    def worker(profile):
        results = []
        for index, checkpoint in enumerate(checkpoints):
            output = args.output / f"{profile}-{index}.json"
            if not output.exists():
                with output.with_suffix(".log").open("w") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "botcolosseo.cli.evaluate_hierarchical_executor",
                            "--checkpoint",
                            str(checkpoint),
                            "--output",
                            str(output),
                            "--profile",
                            profile,
                            "--device",
                            "cuda:0" if profile == "search" else "cuda:1",
                            "--seeds",
                            *map(str, args.seeds),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            report = json.loads(output.read_text())
            expected = {(seed, role) for seed in args.seeds for role in ("host", "opponent")}
            if (
                report["checkpoint_sha256"] != digest(checkpoint)
                or report["profile"] != profile
                or {(case["seed"], case["side"]) for case in report["cases"]} != expected
            ):
                raise ValueError("Upgrade screening protocol or identity mismatch")
            summary = summarize_executor(report)
            row = {"checkpoint": str(checkpoint), "report_sha256": digest(output), **summary}
            results.append(row)
            print(json.dumps({"profile": profile, **row}), flush=True)
        return profile, results

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(worker, ("search", "combat")))
    result = {
        "complete": True,
        "profiles": results,
        "scope": "fixed-command candidate screening; no automatic promotion or D4 pass",
    }
    temporary = args.output / "summary.tmp"
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(args.output / "summary.json")


if __name__ == "__main__":
    main()
