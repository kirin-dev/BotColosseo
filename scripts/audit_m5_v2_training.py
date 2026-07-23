#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.cli.train_ppo import _atomic_json
from botcolosseo.evaluation.m5_v2_training_audit import audit_m5_v2_training


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit an M5 V2 training stage")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--style", choices=("defensive", "explorer"), required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_m5_v2_training(
        args.run_dir,
        style=args.style,
        expected_steps=args.expected_steps,
    )
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        _atomic_json(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
