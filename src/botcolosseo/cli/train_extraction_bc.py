from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from botcolosseo.agents.checkpoint import CheckpointMetadata
from botcolosseo.agents.extraction_model import (
    ExtractionResidualActor,
    create_extraction_actor,
)
from botcolosseo.agents.extraction_teachers import ExtractionStyle
from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.training.bc import (
    BCTrainer,
    DeterministicBatchStream,
    append_jsonl,
    make_validation_loader,
    seed_everything,
)
from botcolosseo.training.extraction_bc import (
    ExtractionChunkDataset,
    load_extraction_shard_paths,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Extraction Strong Base or a residual style branch"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/extraction_v2/training.yaml"),
    )
    parser.add_argument(
        "--style",
        choices=tuple(style.value for style in ExtractionStyle),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--max-train-transitions", type=int)
    parser.add_argument("--max-validation-transitions", type=int)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    return parser


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _config_hash(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _load_base(path: Path, *, scenario_hash: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported Extraction base checkpoint")
    metadata = CheckpointMetadata(**payload["metadata"])
    if metadata.scenario_hash != scenario_hash:
        raise ValueError("Extraction base checkpoint scenario hash does not match")
    actor = create_extraction_actor()
    actor.load_state_dict(payload["model"])
    return actor


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    style = ExtractionStyle(args.style)
    manifest_config = config["manifests"][style.value]
    train_manifest = args.train_manifest or root / manifest_config["train"]
    validation_manifest = (
        args.validation_manifest or root / manifest_config["validation"]
    )
    if not train_manifest.is_absolute():
        train_manifest = root / train_manifest
    if not validation_manifest.is_absolute():
        validation_manifest = root / validation_manifest
    scenario_hash = json.loads(
        (
            root / "assets/scenarios/crystal_run_extraction/manifest.json"
        ).read_text(encoding="utf-8")
    )["wad_sha256"]
    output_dir = args.output_dir or root / config["output_root"] / style.value
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists() and args.resume is None:
        raise FileExistsError(f"Extraction BC output already exists: {metrics_path}")

    base_checkpoint: Path | None = None
    provenance_paths = [config_path, train_manifest, validation_manifest]
    if style is ExtractionStyle.STRONG:
        model = create_extraction_actor()
        default_updates = int(config["strong_updates"])
    else:
        base_checkpoint = args.base_checkpoint or (
            root / config["output_root"] / ExtractionStyle.STRONG.value / "best.pt"
        )
        if not base_checkpoint.is_absolute():
            base_checkpoint = root / base_checkpoint
        if not base_checkpoint.is_file():
            raise FileNotFoundError(
                f"Extraction Strong Base checkpoint is missing: {base_checkpoint}"
            )
        model = ExtractionResidualActor(
            _load_base(base_checkpoint, scenario_hash=scenario_hash)
        )
        provenance_paths.append(base_checkpoint)
        default_updates = int(config["style_updates"])

    total_updates = args.updates or default_updates
    stop_after = args.stop_after or total_updates
    if not 0 < stop_after <= total_updates:
        raise ValueError("--stop-after must be in (0, updates]")
    config_hash = _config_hash(tuple(provenance_paths))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(int(config["seed"]))
    train = ExtractionChunkDataset(
        load_extraction_shard_paths(train_manifest),
        chunk_length=int(config["chunk_length"]),
        max_transitions=args.max_train_transitions,
    )
    validation = ExtractionChunkDataset(
        load_extraction_shard_paths(validation_manifest),
        chunk_length=int(config["chunk_length"]),
        max_transitions=args.max_validation_transitions,
    )
    stream = DeterministicBatchStream(
        train,
        batch_size=int(config["batch_size"]),
        seed=int(config["seed"]),
    )
    validation_loader = make_validation_loader(
        validation,
        batch_size=int(config["batch_size"]),
    )
    trainer = BCTrainer.create(
        model.to(device),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        gradient_clip=float(config["gradient_clip"]),
        total_updates=total_updates,
    )
    if args.resume is not None:
        trainer.load(
            args.resume,
            config_hash=config_hash,
            scenario_hash=scenario_hash,
            restore_rng=True,
        )

    best_loss = math.inf
    best_update: int | None = None
    validation_interval = int(config["validation_interval"])
    latest_validation = None
    while trainer.updates < stop_after:
        metrics = trainer.train_step(stream.batch(trainer.updates))
        if metrics.update == 1 or metrics.update % 25 == 0:
            append_jsonl(metrics_path, {"kind": "train", **asdict(metrics)})
        if metrics.update % validation_interval == 0 or metrics.update == stop_after:
            latest_validation = trainer.validate(validation_loader)
            trainer.save(
                output_dir / "latest.pt",
                config_hash=config_hash,
                scenario_hash=scenario_hash,
            )
            if latest_validation.loss < best_loss:
                best_loss = latest_validation.loss
                best_update = trainer.updates
                trainer.save(
                    output_dir / "best.pt",
                    config_hash=config_hash,
                    scenario_hash=scenario_hash,
                )
            append_jsonl(
                metrics_path,
                {
                    "kind": "validation",
                    "update": trainer.updates,
                    **asdict(latest_validation),
                },
            )
    if latest_validation is None:
        latest_validation = trainer.validate(validation_loader)
    checkpoint = output_dir / "best.pt"
    summary = {
        "base_checkpoint": (
            _display_path(base_checkpoint, root)
            if base_checkpoint is not None
            else None
        ),
        "base_checkpoint_sha256": (
            sha256_file(base_checkpoint) if base_checkpoint is not None else None
        ),
        "best_checkpoint": _display_path(checkpoint, root),
        "best_update": best_update,
        "best_validation_loss": best_loss,
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(config_path.relative_to(root)),
        "config_hash": config_hash,
        "device": str(device),
        "fair_observation_only": True,
        "residual_style_branch": style is not ExtractionStyle.STRONG,
        "scenario_hash": scenario_hash,
        "style": style.value,
        "test_cases_accessed": False,
        "train_manifest_sha256": sha256_file(train_manifest),
        "train_transitions_loaded": train.transition_count,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in trainer.model.parameters()
            if parameter.requires_grad
        ),
        "updates": trainer.updates,
        "validation": asdict(latest_validation),
        "validation_manifest_sha256": sha256_file(validation_manifest),
        "validation_transitions_loaded": validation.transition_count,
    }
    _atomic_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
