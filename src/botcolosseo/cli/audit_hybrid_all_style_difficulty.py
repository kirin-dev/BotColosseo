from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.cli.train_ppo import _atomic_json
from botcolosseo.evaluation.hybrid_all_style_difficulty import (
    audit_hybrid_all_style_difficulty,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the 1,200-row hybrid all-style difficulty matrix"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/m5/hybrid/difficulty-product.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/m5/difficulty/hybrid-all-style-summary.json"),
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        raise FileExistsError("Refusing to overwrite hybrid all-style evidence")
    result = audit_hybrid_all_style_difficulty(
        root,
        config_path=args.config,
    )
    _atomic_json(result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1
