from __future__ import annotations

import os
import random
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from botcolosseo.agents.checkpoint import CheckpointMetadata
from botcolosseo.agents.league_opponents import sha256_file

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class LeagueRunIdentity:
    base_checkpoint_sha256: str
    config_hash: str
    train_manifest_hash: str
    pool_manifest_hash: str
    payoff_report_hash: str
    scenario_hash: str

    def __post_init__(self) -> None:
        if any(
            _SHA256_PATTERN.fullmatch(value) is None
            for value in asdict(self).values()
        ):
            raise ValueError("League run identity requires lowercase SHA-256 values")


@dataclass(frozen=True)
class LeagueCheckpointState:
    environment_steps: int
    updates: int
    episodes: int
    next_pair_slot: int

    def __post_init__(self) -> None:
        if min(
            self.environment_steps,
            self.updates,
            self.episodes,
            self.next_pair_slot,
        ) < 0:
            raise ValueError("League checkpoint counters must be nonnegative")
        if self.next_pair_slot != self.episodes // 2:
            raise ValueError("League checkpoint pair slot does not match episode index")


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


def warm_start_from_m2(
    path: Path,
    model: torch.nn.Module,
    *,
    expected_checkpoint_sha256: str,
    expected_scenario_hash: str,
) -> CheckpointMetadata:
    path = path.expanduser().resolve()
    if sha256_file(path) != expected_checkpoint_sha256:
        raise ValueError("M2 warm-start checkpoint hash does not match")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported M2 warm-start checkpoint schema")
    try:
        metadata = CheckpointMetadata(**payload["metadata"])
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid M2 warm-start metadata") from error
    if metadata.scenario_hash != expected_scenario_hash:
        raise ValueError("M2 warm-start scenario hash does not match")
    try:
        model.load_state_dict(payload["model"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise ValueError("M2 warm-start model dimensions do not match") from error
    return metadata


def save_league_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    identity: LeagueRunIdentity,
    state: LeagueCheckpointState,
) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "identity": asdict(identity),
        "state": asdict(state),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": _rng_state(),
    }
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return path


def load_league_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    expected_identity: LeagueRunIdentity,
    restore_rng: bool = False,
) -> LeagueCheckpointState:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported league checkpoint schema version")
    try:
        identity = LeagueRunIdentity(**payload["identity"])
        state = LeagueCheckpointState(**payload["state"])
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid league checkpoint metadata") from error
    if identity != expected_identity:
        raise ValueError("League checkpoint run identity does not match")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        _restore_rng_state(payload["rng"])
    return state
