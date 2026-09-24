"""Recompute the executor promotion barrier from paired fair reports."""

import argparse
import hashlib
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
    if args.output.exists():
        raise FileExistsError("Preserve an existing promotion decision")
    raw = [path.read_bytes() for path in (args.baseline, args.candidate)]
    reports = [json.loads(content) for content in raw]
    comparison = compare_executors(*reports)
    decision = decide_executor_promotion(comparison, weak_commands=tuple(args.weak_command))
    result = {
        "complete": True,
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "source_sha256": {
            role: hashlib.sha256(content).hexdigest()
            for role, content in zip(("baseline", "candidate"), raw, strict=True)
        },
        "weak_commands": args.weak_command,
        "comparison": comparison,
        "decision": decision,
        "scope": "descriptive screening only; insufficient evidence for formal promotion",
    }
    serialized = json.dumps(result, indent=2, allow_nan=False)
    with args.output.open("x") as stream:
        stream.write(serialized)
    print(json.dumps(decision), flush=True)


if __name__ == "__main__":
    main()
