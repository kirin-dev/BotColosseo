from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.evaluation.checkpoint_release import build_checkpoint_release


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the M6 checkpoint release")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--strong-base", type=Path, required=True)
    parser.add_argument("--aggressive", type=Path, required=True)
    parser.add_argument("--defensive", type=Path, required=True)
    parser.add_argument("--explorer", type=Path, required=True)
    parser.add_argument("--scenario-hash", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_checkpoint_release(
        metrics_path=args.metrics,
        sources={
            "strong_base": args.strong_base,
            "aggressive": args.aggressive,
            "defensive": args.defensive,
            "explorer": args.explorer,
        },
        scenario_hash=args.scenario_hash,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0
