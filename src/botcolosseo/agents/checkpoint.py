from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class CheckpointMetadata:
    config_hash: str
    scenario_hash: str
    counters: dict[str, int]

    def __post_init__(self) -> None:
        if not self.config_hash or not self.scenario_hash:
            raise ValueError("Checkpoint provenance hashes must be nonempty")
        if any(value < 0 for value in self.counters.values()):
            raise ValueError("Checkpoint counters must be nonnegative")


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_training_checkpoint(
    output_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metadata: CheckpointMetadata,
) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    payload = {
        "schema_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metadata": asdict(metadata),
        "rng": _rng_state(),
    }
    try:
        with temporary.open("wb") as target:
            torch.save(payload, target)
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output_path


def load_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    expected_config_hash: str,
    expected_scenario_hash: str,
    restore_rng: bool = False,
) -> CheckpointMetadata:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema version")
    metadata = CheckpointMetadata(**payload["metadata"])
    if metadata.config_hash != expected_config_hash:
        raise ValueError("Checkpoint config hash does not match")
    if metadata.scenario_hash != expected_scenario_hash:
        raise ValueError("Checkpoint scenario hash does not match")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        _restore_rng_state(payload["rng"])
    return metadata
