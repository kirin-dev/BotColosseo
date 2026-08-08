from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from botcolosseo.agents.checkpoint import CheckpointMetadata
from botcolosseo.agents.extraction_model import (
    ExtractionActorCritic,
    ExtractionDefensiveGuardedActor,
    ExtractionResidualStyleActorCritic,
    create_extraction_actor_critic,
)
from botcolosseo.agents.model import RecurrentActor


@dataclass(frozen=True)
class ExtractionStyleRuntimeSpec:
    bottleneck: int
    max_delta: float
    defensive_guardrail: bool
    opportunity_conditioned: bool


def resolve_extraction_style_runtime_spec(
    training_summary: dict[str, object],
    *,
    root: Path,
) -> ExtractionStyleRuntimeSpec:
    """Resolve inference architecture from the training-bound config."""
    config_name = training_summary.get("config")
    if not isinstance(config_name, str) or not config_name:
        raise ValueError("Style training summary has no config provenance")
    root = root.resolve()
    config_path = (root / config_name).resolve()
    try:
        config_path.relative_to(root)
    except ValueError as error:
        raise ValueError("Style training config must be inside the project") from error
    if not config_path.is_file():
        raise FileNotFoundError(f"Style training config is missing: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Style training config must be a mapping")
    bottleneck = training_summary.get(
        "adapter_bottleneck", config.get("adapter_bottleneck")
    )
    max_delta = training_summary.get("max_delta", config.get("max_delta"))
    if (
        not isinstance(bottleneck, int)
        or isinstance(bottleneck, bool)
        or bottleneck <= 0
        or not isinstance(max_delta, (int, float))
        or isinstance(max_delta, bool)
        or not math.isfinite(float(max_delta))
        or float(max_delta) <= 0
    ):
        raise ValueError("Style inference architecture is invalid")
    opportunity_conditioned = training_summary.get("opportunity_conditioning") is True
    guardrail = training_summary.get("inference_guardrail")
    if guardrail is None:
        guardrail = (
            training_summary.get("style") == "defensive"
            and not opportunity_conditioned
        )
    if not isinstance(guardrail, bool):
        raise ValueError("Style inference guardrail flag must be boolean")
    if opportunity_conditioned and guardrail:
        raise ValueError("Opportunity-conditioned styles cannot use a runtime guardrail")
    return ExtractionStyleRuntimeSpec(
        bottleneck=bottleneck,
        max_delta=float(max_delta),
        defensive_guardrail=guardrail,
        opportunity_conditioned=opportunity_conditioned,
    )


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


def validate_extraction_teacher_lineage(
    path: Path,
    *,
    expected_scenario_hash: str,
    expected_teacher_sha256: str,
    expected_sha256: str | None = None,
) -> CheckpointMetadata:
    _, metadata = _training_payload(
        path,
        expected_scenario_hash=expected_scenario_hash,
        expected_sha256=expected_sha256,
    )
    if (
        metadata.lineage.get("teacher_implementation_sha256")
        != expected_teacher_sha256
    ):
        raise ValueError("Extraction BC and PPO Teacher identities do not match")
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
