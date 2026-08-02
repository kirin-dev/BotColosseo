from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from botcolosseo.agents.checkpoint import CheckpointMetadata
from botcolosseo.agents.extraction_model import (
    ExtractionActorCritic,
    ExtractionDefensiveGuardedActor,
    ExtractionResidualStyleActorCritic,
    create_extraction_actor_critic,
)
from botcolosseo.agents.model import RecurrentActor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_payload(
    path: Path,
    *,
    expected_scenario_hash: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, object], CheckpointMetadata]:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("Extraction checkpoint SHA-256 does not match")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported Extraction training checkpoint")
    metadata = CheckpointMetadata(**payload["metadata"])
    if metadata.scenario_hash != expected_scenario_hash:
        raise ValueError("Extraction checkpoint scenario hash does not match")
    if not isinstance(payload.get("model"), dict):
        raise ValueError("Extraction checkpoint has no model state")
    return payload, metadata


def load_extraction_bc_warm_start(
    path: Path,
    model: ExtractionActorCritic,
    *,
    expected_scenario_hash: str,
    expected_sha256: str | None = None,
) -> CheckpointMetadata:
    payload, metadata = _training_payload(
        path,
        expected_scenario_hash=expected_scenario_hash,
        expected_sha256=expected_sha256,
    )
    model.actor.load_state_dict(payload["model"])
    return metadata


def load_extraction_strong_actor(
    path: Path,
    *,
    expected_scenario_hash: str,
    expected_sha256: str | None = None,
    device: torch.device | str = "cpu",
) -> tuple[RecurrentActor, CheckpointMetadata]:
    payload, metadata = _training_payload(
        path,
        expected_scenario_hash=expected_scenario_hash,
        expected_sha256=expected_sha256,
    )
    model = create_extraction_actor_critic()
    model.load_state_dict(payload["model"])
    actor = model.actor.to(device)
    actor.eval()
    return actor, metadata


def load_extraction_strong_actor_critic(
    path: Path,
    *,
    expected_scenario_hash: str,
    expected_sha256: str | None = None,
    device: torch.device | str = "cpu",
) -> tuple[ExtractionActorCritic, CheckpointMetadata]:
    payload, metadata = _training_payload(
        path,
        expected_scenario_hash=expected_scenario_hash,
        expected_sha256=expected_sha256,
    )
    model = create_extraction_actor_critic()
    model.load_state_dict(payload["model"])
    model.to(device)
    return model, metadata


def load_extraction_style_actor(
    path: Path,
    *,
    base_checkpoint: Path,
    expected_scenario_hash: str,
    expected_base_sha256: str,
    bottleneck: int,
    max_delta: float,
    expected_sha256: str | None = None,
    device: torch.device | str = "cpu",
    defensive_guardrail: bool = False,
) -> tuple[torch.nn.Module, CheckpointMetadata]:
    if sha256_file(base_checkpoint) != expected_base_sha256:
        raise ValueError("Extraction style base checkpoint SHA-256 does not match")
    base, _ = load_extraction_strong_actor_critic(
        base_checkpoint,
        expected_scenario_hash=expected_scenario_hash,
        expected_sha256=expected_base_sha256,
        device="cpu",
    )
    frozen_actor = {
        name: value.clone() for name, value in base.actor.state_dict().items()
    }
    payload, metadata = _training_payload(
        path,
        expected_scenario_hash=expected_scenario_hash,
        expected_sha256=expected_sha256,
    )
    model = ExtractionResidualStyleActorCritic(
        base,
        bottleneck=bottleneck,
        max_delta=max_delta,
    )
    model.load_state_dict(payload["model"])
    if any(
        not torch.equal(model.base.actor.state_dict()[name], value)
        for name, value in frozen_actor.items()
    ):
        raise ValueError("Extraction style checkpoint changed the frozen Strong Actor")
    actor: torch.nn.Module = model.public_actor()
    if defensive_guardrail:
        actor = ExtractionDefensiveGuardedActor(actor)
    actor = actor.to(device)
    actor.eval()
    return actor, metadata
