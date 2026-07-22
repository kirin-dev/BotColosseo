from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from botcolosseo.scenarios.league_splits import (
    generate_league_splits,
    write_league_manifests,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate frozen neutral M3 league splits")
    parser.add_argument("--output-root", type=Path, default=Path("configs/m3"))
    parser.add_argument("--master-seed", type=int, default=20260721)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_league_manifests(
        generate_league_splits(master_seed=args.master_seed), args.output_root
    )
    for path in paths:
        print(path)
    return 0
