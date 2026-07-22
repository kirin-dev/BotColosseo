from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from botcolosseo.agents.model import RecurrentActor
from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.scenarios.duel_splits import DuelCase
from botcolosseo.training.bc import (
    BCTrainer,
    BestCheckpointTracker,
    DemonstrationChunkDataset,
    DeterministicBatchStream,
    append_jsonl,
    evaluate_closed_loop_episode,
    load_shard_paths,
    make_validation_loader,
    seed_everything,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the recurrent M2 BC Actor")
    parser.add_argument("--config", type=Path, default=Path("configs/m2/bc.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--max-train-transitions", type=int)
    parser.add_argument("--max-validation-transitions", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    return parser


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _provenance_hash(config: Path, train: Path, validation: Path) -> str:
    digest = hashlib.sha256()
    for path in (config, train, validation):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_validation_case(root: Path) -> DuelCase:
    payload = json.loads((root / "configs/m2/validation.json").read_text())
    return DuelCase(**payload[0])


def _restore_tracker(path: Path) -> tuple[BestCheckpointTracker, set[int]]:
    tracker = BestCheckpointTracker()
    validation_updates: set[int] = set()
    if not path.exists():
        return tracker, validation_updates
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") == "validation":
            update = int(record["update"])
            tracker.update(
                validation_loss=float(record["loss"]),
                objective_rate=float(record["objective_rate"]),
                update=update,
            )
            validation_updates.add(update)
    return tracker, validation_updates


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir or root / config["output_dir"]
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists() and args.resume is None:
        raise FileExistsError(f"BC output already exists: {metrics_path}")
    train_manifest = root / config["train_manifest"]
    validation_manifest = root / config["validation_manifest"]
    config_hash = _provenance_hash(config_path, train_manifest, validation_manifest)
    scenario_hash = json.loads(
        (root / "assets/scenarios/crystal_run/manifest.json").read_text()
    )["wad_sha256"]
    total_updates = args.updates or int(config["updates"])
    stop_after = args.stop_after or total_updates
    if not 0 < stop_after <= total_updates:
        raise ValueError("--stop-after must be in (0, updates]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(int(config["seed"]))
    train = DemonstrationChunkDataset(
        load_shard_paths(train_manifest),
        chunk_length=int(config["chunk_length"]),
        max_transitions=args.max_train_transitions,
    )
    validation = DemonstrationChunkDataset(
        load_shard_paths(validation_manifest),
        chunk_length=int(config["chunk_length"]),
        max_transitions=args.max_validation_transitions,
    )
    stream = DeterministicBatchStream(
        train, batch_size=int(config["batch_size"]), seed=int(config["seed"])
    )
    validation_loader = make_validation_loader(
        validation, batch_size=int(config["batch_size"])
    )
    trainer = BCTrainer.create(
        RecurrentActor().to(device),
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
    tracker, validation_updates = _restore_tracker(metrics_path)
    latest_validation = None
    closed_loop = None
    validation_case = _load_validation_case(root)
    validation_interval = int(config["validation_interval"])

    def validate_and_checkpoint() -> None:
        nonlocal closed_loop, latest_validation
        latest_validation = trainer.validate(validation_loader)
        closed_loop = evaluate_closed_loop_episode(
            trainer.model, root=root, case=validation_case
        )
        objective_rate = float(closed_loop["objective_completed"])
        trainer.save(
            output_dir / "latest.pt",
            config_hash=config_hash,
            scenario_hash=scenario_hash,
        )
        if tracker.update(
            validation_loss=latest_validation.loss,
            objective_rate=objective_rate,
            update=trainer.updates,
        ):
            trainer.save(
                output_dir / "best.pt",
                config_hash=config_hash,
                scenario_hash=scenario_hash,
            )
        append_jsonl(
            metrics_path,
            {
                "kind": "validation",
                "objective_rate": objective_rate,
                "update": trainer.updates,
                **asdict(latest_validation),
            },
        )
        validation_updates.add(trainer.updates)

    if args.resume is not None and trainer.updates not in validation_updates:
        validate_and_checkpoint()
    while trainer.updates < stop_after:
        metrics = trainer.train_step(stream.batch(trainer.updates))
        if metrics.update == 1 or metrics.update % 25 == 0:
            append_jsonl(metrics_path, {"kind": "train", **asdict(metrics)})
        if metrics.update % validation_interval == 0 or metrics.update == stop_after:
            validate_and_checkpoint()
    if closed_loop is not None:
        _atomic_json(closed_loop, output_dir / "closed-loop-validation.json")
    if latest_validation is None:
        latest_validation = trainer.validate(validation_loader)
    summary = {
        "best_checkpoint": str((output_dir / "best.pt").relative_to(root)),
        "best_objective_rate": tracker.best_objective_rate,
        "best_update": tracker.best_update,
        "best_validation_loss": tracker.best_validation_loss,
        "checkpoint_sha256": sha256_file(output_dir / "best.pt"),
        "closed_loop": closed_loop,
        "config": str(config_path.relative_to(root)),
        "config_hash": config_hash,
        "device": str(device),
        "pure_behavioral_cloning": True,
        "scenario_hash": scenario_hash,
        "train_manifest_hash": sha256_file(train_manifest),
        "train_transitions_loaded": train.transition_count,
        "updates": trainer.updates,
        "validation": asdict(latest_validation),
        "validation_manifest_hash": sha256_file(validation_manifest),
        "validation_transitions_loaded": validation.transition_count,
    }
    _atomic_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
