from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.evaluation.user_study import prepare_user_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the blind M6 user study")
    parser.add_argument("--aggressive", type=Path, required=True)
    parser.add_argument("--defensive", type=Path, required=True)
    parser.add_argument("--explorer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assignments", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args(argv)
    manifest = prepare_user_study(
        {
            "aggressive": args.aggressive,
            "defensive": args.defensive,
            "explorer": args.explorer,
        },
        output_dir=args.output_dir,
        assignment_count=args.assignments,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0
