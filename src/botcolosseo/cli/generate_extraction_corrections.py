from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from botcolosseo.agents.extraction_model import load_extraction_policy
from botcolosseo.agents.extraction_policy import ExtractionCheckpointController
from botcolosseo.agents.extraction_teachers import ExtractionStyle
from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.data.extraction_demonstrations import (
    generate_extraction_demonstrations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect DAgger correction states for Extraction v2"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/extraction_v2/demonstrations.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--style",
        choices=tuple(style.value for style in ExtractionStyle),
        required=True,
    )
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--transitions", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = (
        args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    scenario_hash = json.loads(
        (
            root / "assets/scenarios/crystal_run_extraction/manifest.json"
        ).read_text(encoding="utf-8")
    )["wad_sha256"]
    model, _ = load_extraction_policy(
        checkpoint,
        style=args.style,
        scenario_hash=scenario_hash,
        device=device,
    )
    controller = ExtractionCheckpointController(model, device=device)
    cases_path = root / config[f"{args.split}_cases"]
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    )
    result = generate_extraction_demonstrations(
        root=root,
        split=args.split,
        cases_path=cases_path,
        output_dir=output_dir,
        style=args.style,
        transitions=args.transitions,
        shard_size=args.shard_size,
        max_decisions=int(config["max_decisions"]),
        rollout_controller=controller,
        source_policy_sha256=sha256_file(checkpoint),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
