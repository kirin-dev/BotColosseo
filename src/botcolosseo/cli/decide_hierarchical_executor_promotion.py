"""Recompute the executor promotion barrier from paired fair reports."""

import argparse
import json
from pathlib import Path

from botcolosseo.evaluation.hierarchical_executor import (
    compare_executors,
    decide_executor_promotion,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--weak-command", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = compare_executors(
        json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text())
    )
    decision = decide_executor_promotion(comparison, weak_commands=tuple(args.weak_command))
    result = {
        "complete": True,
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "weak_commands": args.weak_command,
        "comparison": comparison,
        "decision": decision,
        "scope": "paired descriptive barrier; no automatic checkpoint promotion",
    }
    if args.output.exists():
        raise FileExistsError("Preserve an existing promotion decision")
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(args.output)
    print(json.dumps(decision), flush=True)


if __name__ == "__main__":
    main()
