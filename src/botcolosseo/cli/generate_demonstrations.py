from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from botcolosseo.data.demonstrations import (
    generate_demonstration_split,
    render_demonstration_distribution,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic M2 demonstrations")
    parser.add_argument("--config", type=Path, default=Path("configs/m2/demonstrations.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/m2"))
    parser.add_argument("--split", choices=("train", "validation", "all"), default="all")
    parser.add_argument("--transitions", type=int)
    parser.add_argument("--shard-size", type=int)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/m2/demonstrations-manifest.json"),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("docs/assets/m2-demonstration-distribution.png"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    selected = ("train", "validation") if args.split == "all" else (args.split,)
    manifests = []
    for split in selected:
        split_config = config["splits"][split]
        cases_path = root / split_config["cases"]
        manifests.append(
            generate_demonstration_split(
                root=root,
                split=split,
                cases_path=cases_path,
                output_dir=output_dir / split,
                transitions=args.transitions or int(split_config["transitions"]),
                shard_size=args.shard_size or int(config["shard_size"]),
                case_transition_cap=int(config["case_transition_cap"]),
            )
        )
    report = {
        "config": str(config_path.relative_to(root)),
        "schema_version": 1,
        "splits": manifests,
        "test_cases_accessed": False,
    }
    report_path = args.report if args.report.is_absolute() else root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_path = args.plot if args.plot.is_absolute() else root / args.plot
    render_demonstration_distribution(manifests, plot_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
