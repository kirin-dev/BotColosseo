from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from botcolosseo.agents.duel_teachers import EXPLORER_ROUTE_CYCLE
from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.agents.model import AsymmetricActorCritic
from botcolosseo.agents.style_model import StyledActorCritic
from botcolosseo.cli.train_ppo import _atomic_json
from botcolosseo.training.bc import (
    DemonstrationChunkDataset,
    DeterministicBatchStream,
    append_jsonl,
    load_shard_paths,
    seed_everything,
)
from botcolosseo.training.style_distillation import (
    StyleDistillationTrainer,
    evaluate_explorer_distillation,
    save_style_distillation_checkpoint,
    save_style_neutral_checkpoint,
)

_CONFIG_FIELDS = {
    "schema_version",
    "seed",
    "train_cases",
    "train_manifest",
    "transitions",
    "shard_size",
    "case_transition_cap",
    "min_route_transitions",
    "min_route_fraction",
    "max_route_windows_per_route",
    "chunk_length",
    "batch_size",
    "updates",
    "learning_rate",
    "weight_decay",
    "gradient_clip",
    "beta_kl",
    "bottleneck",
}
_DATA_GATES = {
    "complete",
    "route_transitions",
    "route_coverage",
    "route_balance",
    "opponent_coverage",
    "side_coverage",
    "context_balance",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain the Explorer style adapter")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/m5/explorer_distillation.yaml")
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int)
    return parser


def _validate_config(config: dict[str, Any]) -> None:
    if set(config) != _CONFIG_FIELDS or config.get("schema_version") != 1:
        raise ValueError("Explorer distillation config does not match schema version 1")
    nonpositive_allowed = {"schema_version", "train_cases", "train_manifest", "weight_decay"}
    if any(float(config[name]) <= 0 for name in _CONFIG_FIELDS - nonpositive_allowed):
        raise ValueError("Explorer distillation config values must be positive")
    if float(config["weight_decay"]) < 0:
        raise ValueError("Explorer distillation weight decay must be nonnegative")
    if Path(str(config["train_cases"])).name != "train.json":
        raise ValueError("Explorer distillation must use training cases")


def _load_base(path: Path, *, root: Path) -> tuple[AsymmetricActorCritic, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    identity = payload.get("identity")
    if payload.get("schema_version") != 1 or not isinstance(identity, dict):
        raise ValueError("Explorer distillation base must be an M3 checkpoint")
    scenario_hash = identity.get("scenario_hash")
    if not isinstance(scenario_hash, str):
        raise ValueError("M3 checkpoint has no scenario identity")
    manifest = json.loads(
        (root / "assets/scenarios/crystal_run/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if scenario_hash != manifest.get("wad_sha256"):
        raise ValueError("M3 checkpoint scenario does not match the repository")
    model = AsymmetricActorCritic()
    model.load_state_dict(payload["model"], strict=True)
    return model, scenario_hash


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Explorer distillation config must be a mapping")
    _validate_config(config)
    updates = int(config["updates"]) if args.updates is None else args.updates
    if updates <= 0:
        raise ValueError("Explorer distillation updates must be positive")
    base_path = (
        args.base_checkpoint
        if args.base_checkpoint.is_absolute()
        else root / args.base_checkpoint
    )
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(f"Explorer distillation output exists: {metrics_path}")
    manifest_path = root / str(config["train_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_hash = sha256_file(base_path)
    gates = manifest.get("gate")
    route_windows = manifest.get("route_window_counts")
    if (
        manifest.get("style") != "explorer"
        or manifest.get("split") != "train"
        or manifest.get("passed") is not True
        or manifest.get("test_cases_accessed") is not False
        or manifest.get("base_checkpoint_sha256") != base_hash
        or manifest.get("transitions") != int(config["transitions"])
        or not isinstance(gates, dict)
        or set(gates) != _DATA_GATES
        or not all(value is True for value in gates.values())
        or not isinstance(route_windows, dict)
        or set(route_windows) != set(EXPLORER_ROUTE_CYCLE)
        or not all(int(route_windows[route]) > 0 for route in EXPLORER_ROUTE_CYCLE)
    ):
        raise ValueError("Explorer distillation manifest is not eligible for training")
    base, scenario_hash = _load_base(base_path, root=root)
    if manifest.get("scenario_hash") != scenario_hash:
        raise ValueError("Explorer demonstrations and M3 base use different scenarios")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(int(config["seed"]))
    model = StyledActorCritic.from_base(
        base, bottleneck=int(config["bottleneck"])
    ).to(device)
    manifest_hash = sha256_file(manifest_path)
    config_hash = sha256_file(config_path)
    neutral = save_style_neutral_checkpoint(
        run_dir / "neutral.pt",
        model=model,
        style="explorer",
        base_checkpoint_sha256=base_hash,
        scenario_hash=scenario_hash,
        data_manifest_sha256=manifest_hash,
        config_hash=config_hash,
    )
    dataset = DemonstrationChunkDataset(
        load_shard_paths(manifest_path),
        chunk_length=int(config["chunk_length"]),
        allow_masked=True,
    )
    stream = DeterministicBatchStream(
        dataset,
        batch_size=int(config["batch_size"]),
        seed=int(config["seed"]),
    )
    trainer = StyleDistillationTrainer.create(
        model,
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        gradient_clip=float(config["gradient_clip"]),
        beta_kl=float(config["beta_kl"]),
        total_updates=updates,
    )
    last = None
    for update in range(updates):
        last = trainer.train_step(stream.batch(update))
        if not all(
            math.isfinite(value)
            for value in asdict(last).values()
            if isinstance(value, float)
        ):
            raise FloatingPointError("Explorer distillation produced non-finite metrics")
        if update == 0 or trainer.updates % 25 == 0 or trainer.updates == updates:
            append_jsonl(metrics_path, asdict(last))
            print(json.dumps(asdict(last), sort_keys=True), flush=True)
    if last is None:
        raise RuntimeError("Explorer distillation completed no updates")
    checkpoint = save_style_distillation_checkpoint(
        run_dir / "style-pretrained.pt",
        model=model,
        base_checkpoint_sha256=base_hash,
        scenario_hash=scenario_hash,
        data_manifest_sha256=manifest_hash,
        config_hash=config_hash,
        updates=trainer.updates,
        style="explorer",
    )
    offline = evaluate_explorer_distillation(
        model,
        DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
        ),
    )
    summary = {
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": base_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "completed": trainer.updates == updates,
        "config": str(config_path),
        "config_hash": config_hash,
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": manifest_hash,
        "device": str(device),
        "final_metrics": asdict(last),
        "frozen_base": True,
        "neutral_checkpoint": str(neutral),
        "neutral_checkpoint_sha256": sha256_file(neutral),
        "offline_evaluation": offline,
        "passed": trainer.updates == updates and bool(offline["passed"]),
        "scenario_hash": scenario_hash,
        "style": "explorer",
        "test_cases_accessed": False,
        "updates": trainer.updates,
    }
    _atomic_json(summary, run_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["passed"] else 1
