from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from botcolosseo.agents.league_opponents import sha256_file

_STYLE_PREFIXES = ("adapter.", "policy.")
_FROZEN_ACTOR_PREFIX = "base.actor."


def _model_state(payload: dict[str, Any], *, label: str) -> dict[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{label} checkpoint has no model state")
    if any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError(f"{label} checkpoint model state is invalid")
    return state


def _interpolation_hash(*, distilled_sha256: str, ppo_sha256: str, alpha: float) -> str:
    payload = json.dumps(
        {
            "alpha": alpha,
            "distilled_checkpoint_sha256": distilled_sha256,
            "ppo_checkpoint_sha256": ppo_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _neutral_interpolation_hash(
    *, neutral_sha256: str, distilled_sha256: str, alpha: float
) -> str:
    payload = json.dumps(
        {
            "alpha": alpha,
            "distilled_checkpoint_sha256": distilled_sha256,
            "neutral_checkpoint_sha256": neutral_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def interpolate_style_checkpoints(
    distilled_path: Path,
    ppo_path: Path,
    output_path: Path,
    *,
    alpha: float,
) -> dict[str, object]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("Style interpolation alpha must be strictly between 0 and 1")
    distilled_path = distilled_path.expanduser().resolve()
    ppo_path = ppo_path.expanduser().resolve()
    for path in (distilled_path, ppo_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    distilled = torch.load(distilled_path, map_location="cpu", weights_only=False)
    ppo = torch.load(ppo_path, map_location="cpu", weights_only=False)
    if (
        distilled.get("schema_version") != 1
        or distilled.get("kind") != "style_distillation"
        or distilled.get("style") != "aggressive"
    ):
        raise ValueError("Distilled source is not an Aggressive style checkpoint")
    identity = ppo.get("identity")
    if ppo.get("schema_version") != 1 or not isinstance(identity, dict):
        raise ValueError("PPO source is not a league checkpoint")
    base_hash = distilled.get("base_checkpoint_sha256")
    scenario_hash = distilled.get("scenario_hash")
    if (
        not isinstance(base_hash, str)
        or identity.get("base_checkpoint_sha256") != base_hash
        or not isinstance(scenario_hash, str)
        or identity.get("scenario_hash") != scenario_hash
    ):
        raise ValueError("Style interpolation source identities do not match")
    distilled_state = _model_state(distilled, label="Distilled")
    ppo_state = _model_state(ppo, label="PPO")
    if set(distilled_state) != set(ppo_state):
        raise ValueError("Style interpolation model structures do not match")
    style_names = tuple(name for name in distilled_state if name.startswith(_STYLE_PREFIXES))
    if not style_names:
        raise ValueError("Style interpolation sources contain no style branch")
    frozen_names = tuple(name for name in distilled_state if name.startswith(_FROZEN_ACTOR_PREFIX))
    if any(not torch.equal(distilled_state[name], ppo_state[name]) for name in frozen_names):
        raise ValueError("PPO source changed the frozen Strong Base actor")
    output_state = {name: value.detach().clone() for name, value in distilled_state.items()}
    for name in style_names:
        left = ppo_state[name]
        right = distilled_state[name]
        if left.shape != right.shape or not (
            left.is_floating_point() and right.is_floating_point()
        ):
            raise ValueError("Style interpolation tensors are incompatible")
        output_state[name] = torch.lerp(left, right, alpha)
    distilled_hash = sha256_file(distilled_path)
    ppo_hash = sha256_file(ppo_path)
    payload = {
        "schema_version": 1,
        "kind": "style_interpolation",
        "style": "aggressive",
        "alpha": alpha,
        "base_checkpoint_sha256": base_hash,
        "scenario_hash": scenario_hash,
        "distilled_checkpoint_sha256": distilled_hash,
        "ppo_checkpoint_sha256": ppo_hash,
        "interpolation_sha256": _interpolation_hash(
            distilled_sha256=distilled_hash,
            ppo_sha256=ppo_hash,
            alpha=alpha,
        ),
        "model": output_state,
    }
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output_path)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return {
        "alpha": alpha,
        "base_checkpoint_sha256": base_hash,
        "checkpoint": str(output_path),
        "checkpoint_sha256": sha256_file(output_path),
        "distilled_checkpoint_sha256": distilled_hash,
        "frozen_base_actor": True,
        "interpolation_sha256": payload["interpolation_sha256"],
        "ppo_checkpoint_sha256": ppo_hash,
        "scenario_hash": scenario_hash,
        "style": "aggressive",
        "test_cases_accessed": False,
    }


def interpolate_defensive_checkpoints(
    neutral_path: Path,
    distilled_path: Path,
    output_path: Path,
    *,
    alpha: float,
) -> dict[str, object]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("Style interpolation alpha must be strictly between 0 and 1")
    neutral_path = neutral_path.expanduser().resolve()
    distilled_path = distilled_path.expanduser().resolve()
    for path in (neutral_path, distilled_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    neutral = torch.load(neutral_path, map_location="cpu", weights_only=False)
    distilled = torch.load(distilled_path, map_location="cpu", weights_only=False)
    if (
        neutral.get("schema_version") != 1
        or neutral.get("kind") != "style_neutral"
        or neutral.get("style") != "defensive"
        or neutral.get("updates") != 0
    ):
        raise ValueError("Neutral source is not a Defensive style checkpoint")
    if (
        distilled.get("schema_version") != 1
        or distilled.get("kind") != "style_distillation"
        or distilled.get("style") != "defensive"
        or not isinstance(distilled.get("updates"), int)
        or int(distilled["updates"]) <= 0
    ):
        raise ValueError("Distilled source is not a Defensive style checkpoint")
    identity_fields = (
        "base_checkpoint_sha256",
        "scenario_hash",
        "data_manifest_sha256",
        "config_hash",
    )
    if any(neutral.get(name) != distilled.get(name) for name in identity_fields):
        raise ValueError("Defensive interpolation source identities do not match")
    neutral_state = _model_state(neutral, label="Neutral")
    distilled_state = _model_state(distilled, label="Distilled")
    if set(neutral_state) != set(distilled_state):
        raise ValueError("Defensive interpolation model structures do not match")
    style_names = tuple(name for name in neutral_state if name.startswith(_STYLE_PREFIXES))
    if not style_names:
        raise ValueError("Defensive interpolation sources contain no style branch")
    frozen_names = tuple(name for name in neutral_state if name.startswith("base."))
    if any(not torch.equal(neutral_state[name], distilled_state[name]) for name in frozen_names):
        raise ValueError("Distillation changed the frozen Strong Base")
    output_state = {name: value.detach().clone() for name, value in neutral_state.items()}
    for name in style_names:
        left = neutral_state[name]
        right = distilled_state[name]
        if left.shape != right.shape or not (
            left.is_floating_point() and right.is_floating_point()
        ):
            raise ValueError("Defensive interpolation tensors are incompatible")
        output_state[name] = torch.lerp(left, right, alpha)
    neutral_hash = sha256_file(neutral_path)
    distilled_hash = sha256_file(distilled_path)
    interpolation_hash = _neutral_interpolation_hash(
        neutral_sha256=neutral_hash,
        distilled_sha256=distilled_hash,
        alpha=alpha,
    )
    payload = {
        "schema_version": 1,
        "kind": "style_interpolation",
        "style": "defensive",
        "alpha": alpha,
        "base_checkpoint_sha256": neutral["base_checkpoint_sha256"],
        "scenario_hash": neutral["scenario_hash"],
        "distilled_checkpoint_sha256": distilled_hash,
        "neutral_checkpoint_sha256": neutral_hash,
        "interpolation_sha256": interpolation_hash,
        "model": output_state,
    }
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output_path)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return {
        "alpha": alpha,
        "base_checkpoint_sha256": neutral["base_checkpoint_sha256"],
        "checkpoint": str(output_path),
        "checkpoint_sha256": sha256_file(output_path),
        "distilled_checkpoint_sha256": distilled_hash,
        "frozen_base": True,
        "interpolation_sha256": interpolation_hash,
        "neutral_checkpoint_sha256": neutral_hash,
        "scenario_hash": neutral["scenario_hash"],
        "style": "defensive",
        "test_cases_accessed": False,
    }
