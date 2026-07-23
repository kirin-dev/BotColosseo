from __future__ import annotations

import argparse
from pathlib import Path

from botcolosseo.evaluation.m6_showcase_config import (
    build_m6_showcase_config,
    write_m6_showcase_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the hash-bound M6 showcase config"
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("reports/m6/showcase-metrics.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/showcase/m6.yaml"),
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    metrics = args.metrics if args.metrics.is_absolute() else root / args.metrics
    output = args.output if args.output.is_absolute() else root / args.output
    payload = build_m6_showcase_config(root=root, metrics_path=metrics)
    written = write_m6_showcase_config(payload, output)
    print(written.relative_to(root))
    return 0
