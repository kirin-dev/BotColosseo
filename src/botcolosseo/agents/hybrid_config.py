from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from botcolosseo.agents.style_governor import (
    DefensiveGovernorConfig,
    ExplorerGovernorConfig,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "candidate_id",
    "style",
    "base_checkpoint",
    "base_checkpoint_sha256",
    "scenario_hash",
    "test_cases_accessed",
    "governor",
}
_DEFENSIVE_FIELDS = {
    "guard_decisions",
    "guard_bias",
    "disengage_decisions",
    "disengage_bias",
    "recover_decisions",
    "low_health_threshold",
    "health_drop_threshold",
    "max_consecutive_interventions",
}
_EXPLORER_FIELDS = {
    "route_decisions",
    "route_bias",
    "flank_bias",
    "stall_repeat_decisions",
    "stall_recovery_decisions",
    "low_health_threshold",
    "max_consecutive_interventions",
}


@dataclass(frozen=True)
class HybridPolicyConfig:
    candidate_id: str
    style: Literal["defensive", "explorer"]
    base_checkpoint: Path
    base_checkpoint_sha256: str
    scenario_hash: str
    test_cases_accessed: Literal[False]
    governor: DefensiveGovernorConfig | ExplorerGovernorConfig
    config_sha256: str


def load_hybrid_policy_config(path: Path, *, root: Path) -> HybridPolicyConfig:
    payload_bytes = path.read_bytes()
    payload = yaml.safe_load(payload_bytes)
    if not isinstance(payload, dict) or set(payload) != _TOP_LEVEL_FIELDS:
        raise ValueError("Hybrid config fields do not match schema")
    if payload["schema_version"] != 1:
        raise ValueError("Hybrid config requires schema_version 1")
    candidate_id = payload["candidate_id"]
    if (
        not isinstance(candidate_id, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9._-]*", candidate_id) is None
    ):
        raise ValueError("Hybrid candidate_id is invalid")
    style = payload["style"]
    if style not in ("defensive", "explorer"):
        raise ValueError("Hybrid style must be defensive or explorer")
    checkpoint = payload["base_checkpoint"]
    checkpoint_sha256 = payload["base_checkpoint_sha256"]
    scenario_hash = payload["scenario_hash"]
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("Hybrid Base checkpoint path is invalid")
    if not isinstance(checkpoint_sha256, str) or _SHA256.fullmatch(checkpoint_sha256) is None:
        raise ValueError("Hybrid Base checkpoint SHA-256 is invalid")
    if not isinstance(scenario_hash, str) or _SHA256.fullmatch(scenario_hash) is None:
        raise ValueError("Hybrid scenario hash is invalid")
    if payload["test_cases_accessed"] is not False:
        raise ValueError("Hybrid candidate config must not access test cases")
    governor_payload = payload["governor"]
    if not isinstance(governor_payload, dict):
        raise ValueError("Hybrid governor config must be a mapping")
    try:
        if style == "defensive":
            if set(governor_payload) != _DEFENSIVE_FIELDS:
                raise ValueError("Defensive governor fields do not match schema")
            governor = DefensiveGovernorConfig(**governor_payload)
        else:
            if set(governor_payload) != _EXPLORER_FIELDS:
                raise ValueError("Explorer governor fields do not match schema")
            governor = ExplorerGovernorConfig(**governor_payload)
    except TypeError as error:
        raise ValueError("Hybrid governor value types are invalid") from error
    resolved_checkpoint = Path(checkpoint)
    if not resolved_checkpoint.is_absolute():
        resolved_checkpoint = (root / resolved_checkpoint).resolve()
    return HybridPolicyConfig(
        candidate_id=candidate_id,
        style=style,
        base_checkpoint=resolved_checkpoint,
        base_checkpoint_sha256=checkpoint_sha256,
        scenario_hash=scenario_hash,
        test_cases_accessed=False,
        governor=governor,
        config_sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )
