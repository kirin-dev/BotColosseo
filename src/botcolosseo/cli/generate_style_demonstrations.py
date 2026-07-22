from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from botcolosseo.data.style_demonstrations import generate_aggressive_demonstrations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate success-filtered Aggressive demonstrations"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/m4/aggressive_distillation.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/m4/aggressive"))
    parser.add_argument("--transitions", type=int)
    parser.add_argument("--shard-size", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = generate_aggressive_demonstrations(
        root=root,
        cases_path=root / config["train_cases"],
        output_dir=output_dir,
        transitions=args.transitions or int(config["transitions"]),
        shard_size=args.shard_size or int(config["shard_size"]),
        case_transition_cap=int(config["case_transition_cap"]),
        non_attack_stride=int(config["non_attack_stride"]),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0
