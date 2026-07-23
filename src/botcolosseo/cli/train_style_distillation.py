from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

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
    evaluate_style_distillation,
    save_style_distillation_checkpoint,
)

_CONFIG_FIELDS = {
    "schema_version",
    "seed",
    "train_cases",
    "train_manifest",
    "transitions",
    "shard_size",
    "case_transition_cap",
    "non_attack_stride",
    "chunk_length",
    "batch_size",
    "updates",
    "learning_rate",
    "weight_decay",
    "gradient_clip",
    "beta_kl",
    "bottleneck",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain the Aggressive style adapter")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/m4/aggressive_distillation.yaml")
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int)
    return parser


def _validate_config(config: dict[str, Any]) -> None:
    if set(config) != _CONFIG_FIELDS or config.get("schema_version") != 1:
        raise ValueError("Aggressive distillation config does not match schema version 1")
    positive = _CONFIG_FIELDS - {"schema_version", "train_cases", "train_manifest", "weight_decay"}
    if any(float(config[name]) <= 0 for name in positive):
        raise ValueError("Aggressive distillation config values must be positive")
    if float(config["weight_decay"]) < 0:
        raise ValueError("Aggressive distillation weight decay must be nonnegative")
    if Path(str(config["train_cases"])).name != "train.json":
        raise ValueError("Aggressive distillation must use training cases")


def _load_base(path: Path, *, root: Path) -> tuple[AsymmetricActorCritic, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    identity = payload.get("identity")
    if payload.get("schema_version") != 1 or not isinstance(identity, dict):
        raise ValueError("Aggressive distillation base must be an M3 checkpoint")
    scenario_hash = identity.get("scenario_hash")
    if not isinstance(scenario_hash, str):
        raise ValueError("M3 checkpoint has no scenario identity")
    manifest = json.loads(
        (root / "assets/scenarios/crystal_run/manifest.json").read_text(encoding="utf-8")
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
        raise ValueError("Aggressive distillation config must be a mapping")
    _validate_config(config)
    updates = args.updates or int(config["updates"])
    if updates <= 0:
        raise ValueError("Aggressive distillation updates must be positive")
    base_path = (
        args.base_checkpoint if args.base_checkpoint.is_absolute() else root / args.base_checkpoint
    )
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(f"Style distillation output already exists: {metrics_path}")
    manifest_path = root / str(config["train_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("style") != "aggressive"
        or manifest.get("split") != "train"
        or manifest.get("test_cases_accessed") is not False
        or int(manifest.get("label_counts", {}).get("successful_attack", 0)) <= 0
    ):
        raise ValueError("Aggressive distillation manifest is not eligible for training")
    base_hash = sha256_file(base_path)
    base, scenario_hash = _load_base(base_path, root=root)
    if manifest.get("scenario_hash") != scenario_hash:
        raise ValueError("Aggressive demonstrations and M3 base use different scenarios")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(int(config["seed"]))
    model = StyledActorCritic.from_base(base, bottleneck=int(config["bottleneck"])).to(device)
    dataset = DemonstrationChunkDataset(
        load_shard_paths(manifest_path),
        chunk_length=int(config["chunk_length"]),
        allow_masked=True,
    )
    stream = DeterministicBatchStream(
        dataset, batch_size=int(config["batch_size"]), seed=int(config["seed"])
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
        if update == 0 or trainer.updates % 25 == 0 or trainer.updates == updates:
            append_jsonl(metrics_path, asdict(last))
            print(json.dumps(asdict(last), sort_keys=True), flush=True)
    if last is None:
        raise RuntimeError("Aggressive distillation completed no updates")
    checkpoint = save_style_distillation_checkpoint(
        run_dir / "style-pretrained.pt",
        model=model,
        base_checkpoint_sha256=base_hash,
        scenario_hash=scenario_hash,
        data_manifest_sha256=sha256_file(manifest_path),
        config_hash=sha256_file(config_path),
        updates=trainer.updates,
    )
    offline_evaluation = evaluate_style_distillation(
        model,
        DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False),
    )
    summary = {
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": base_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "completed": trainer.updates == updates,
        "config": str(config_path),
        "config_hash": sha256_file(config_path),
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": sha256_file(manifest_path),
        "device": str(device),
        "final_metrics": asdict(last),
        "frozen_base": True,
        "offline_evaluation": offline_evaluation,
        "scenario_hash": scenario_hash,
        "style": "aggressive",
        "test_cases_accessed": False,
        "updates": trainer.updates,
    }
    _atomic_json(summary, run_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0
