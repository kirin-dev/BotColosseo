from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from botcolosseo.demo.m2_training_plot import (
    load_training_evidence,
    render_training_plot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render tracked M2 training evidence")
    parser.add_argument(
        "--bc-summary",
        type=Path,
        default=Path("reports/m2/bc-training-summary.json"),
    )
    parser.add_argument(
        "--ppo-summary",
        type=Path,
        default=Path("reports/m2/ppo-training-summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/m2-training-curves.png"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evidence = load_training_evidence(args.bc_summary, args.ppo_summary)
    output = render_training_plot(evidence, args.output)
    print(output)
    return 0
