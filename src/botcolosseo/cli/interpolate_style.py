from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.cli.train_ppo import _atomic_json
from botcolosseo.training.style_interpolation import interpolate_style_checkpoints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interpolate two Aggressive style checkpoints")
    parser.add_argument("--distilled-checkpoint", type=Path, required=True)
    parser.add_argument("--ppo-checkpoint", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else root / path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    report = interpolate_style_checkpoints(
        _resolve(root, args.distilled_checkpoint),
        _resolve(root, args.ppo_checkpoint),
        _resolve(root, args.output),
        alpha=args.alpha,
    )
    _atomic_json(report, _resolve(root, args.report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
