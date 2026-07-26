from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from botcolosseo.agents.extraction_teachers import ExtractionStyle
from botcolosseo.data.extraction_demonstrations import (
    generate_extraction_demonstrations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fair-observation Crystal Run: Extraction demonstrations"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/extraction_v2/demonstrations.yaml"),
    )
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument(
        "--style",
        choices=tuple(style.value for style in ExtractionStyle),
        required=True,
    )
    parser.add_argument("--transitions", type=int)
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cases_path = root / config[f"{args.split}_cases"]
    transitions = args.transitions or int(config[f"{args.split}_transitions"])
    shard_size = args.shard_size or int(config["shard_size"])
    output_dir = args.output_dir or (
        root / config["output_root"] / args.style / args.split
    )
    result = generate_extraction_demonstrations(
        root=root,
        split=args.split,
        cases_path=cases_path,
        output_dir=output_dir,
        style=args.style,
        transitions=transitions,
        shard_size=shard_size,
        max_decisions=int(config["max_decisions"]),
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
