from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from botcolosseo.agents.league_opponents import (
    CheckpointOpponentPolicy,
    OpponentSpec,
    sha256_file,
)
from botcolosseo.data.defensive_demonstrations import (
    generate_defensive_demonstrations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate risk-conditioned Defensive demonstrations"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/m5/defensive_distillation.yaml")
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/generated/m5/defensive")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transitions", type=int)
    parser.add_argument("--shard-size", type=int)
    return parser


def _positive_int(config: dict[str, Any], name: str, override: int | None = None) -> int:
    value = int(config[name]) if override is None else override
    if value <= 0:
        raise ValueError(f"Defensive demonstration {name} must be positive")
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Defensive demonstration config must use schema version 1")
    cases_path = root / str(config["train_cases"])
    if cases_path.name != "train.json":
        raise ValueError("Defensive demonstrations must use training cases")
    base_path = (
        args.base_checkpoint
        if args.base_checkpoint.is_absolute()
        else root / args.base_checkpoint
    )
    scenario_manifest = json.loads(
        (root / "assets/scenarios/crystal_run/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    scenario_hash = scenario_manifest.get("wad_sha256")
    if not isinstance(scenario_hash, str) or not scenario_hash:
        raise ValueError("Crystal Run manifest has no scenario hash")
    spec = OpponentSpec(
        opponent_id="strong_base",
        kind="checkpoint",
        checkpoint=str(base_path),
        checkpoint_sha256=sha256_file(base_path),
        scenario_hash=scenario_hash,
        selection_evidence="m3:strong-base",
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    base_policy = CheckpointOpponentPolicy.load(spec, device=device)
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest = generate_defensive_demonstrations(
        root=root,
        cases_path=cases_path,
        output_dir=output_dir,
        base_policy=base_policy,
        transitions=_positive_int(config, "transitions", args.transitions),
        shard_size=_positive_int(config, "shard_size", args.shard_size),
        case_transition_cap=_positive_int(config, "case_transition_cap"),
        min_risk_transitions=_positive_int(config, "min_risk_transitions"),
        min_denial_recovery_windows=_positive_int(
            config, "min_denial_recovery_windows"
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0 if manifest["passed"] else 1
