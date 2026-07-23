#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.cli.train_ppo import _atomic_json
from botcolosseo.evaluation.m5_v2_decision import decide_m5_v2_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply the frozen M5 V2 50k rule")
    parser.add_argument("--style", choices=("defensive", "explorer"), required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = decide_m5_v2_candidate(
        style=args.style,
        training_summary=args.training_summary,
        candidate=args.candidate,
        smoke_dir=args.smoke_dir,
    )
    _atomic_json(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
