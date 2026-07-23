from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.cli.train_ppo import _atomic_json
from botcolosseo.evaluation.all_style_difficulty_audit import (
    audit_all_style_difficulty,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit combined M5 all-style difficulty evidence"
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/m5/difficulty/all-style-summary.json"),
    )
    args = parser.parse_args(argv)
    root = (args.root or Path(__file__).resolve().parents[3]).resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    result = audit_all_style_difficulty(root)
    if output.exists():
        raise FileExistsError("Refusing to overwrite all-style difficulty evidence")
    _atomic_json(result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1
