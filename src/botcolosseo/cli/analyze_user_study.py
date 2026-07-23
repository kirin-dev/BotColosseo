from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.evaluation.user_study import analyze_user_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze anonymous M6 responses")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze_user_study(args.package_dir, args.responses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
