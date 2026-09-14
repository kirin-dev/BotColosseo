"""Validated, episode-local command demonstration sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from botcolosseo.data.extraction_demonstrations import EXTRACTION_SCALAR_DIM
from botcolosseo.training.hierarchical_protocol import COMMAND_SCHEMA, Command


def load_command_episode(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if str(data["command_schema"]) != COMMAND_SCHEMA:
        raise ValueError("Incompatible command schema")
    count = len(data["actions"])
    shapes = {"frames": (count, 84, 84), "scalars": (count, EXTRACTION_SCALAR_DIM)}
    shapes.update(
        {
            key: (count,)
            for key in (
                "actions",
                "previous_actions",
                "commands",
                "valid",
                "episode_start",
                "command_start",
                "difficulty",
            )
        }
    )
    if not count or any(data[key].shape != shape for key, shape in shapes.items()):
        raise ValueError("Invalid trajectory dimensions")
    if data["frames"].dtype != np.uint8 or not np.isfinite(data["scalars"]).all():
        raise ValueError("Invalid fair observation data")
    if data["scalars"].dtype != np.float32 or data["difficulty"].dtype != np.float32:
        raise ValueError("Continuous inputs must use float32")
    for key in ("valid", "episode_start", "command_start"):
        if data[key].dtype != np.bool_:
            raise ValueError(f"{key} must be boolean")
    for key, maximum in (("commands", len(Command)), ("actions", 13), ("previous_actions", 13)):
        if data[key].dtype != np.int64 or not ((data[key] >= 0) & (data[key] < maximum)).all():
            raise ValueError(f"Invalid {key}")
    if not data["episode_start"][0] or data["episode_start"][1:].any():
        raise ValueError("File must contain exactly one episode")
    changes = np.r_[True, data["commands"][1:] != data["commands"][:-1]]
    if not data["command_start"][changes].all():
        raise ValueError("Command changes require boundary records")
    if not ((data["difficulty"] >= 0) & (data["difficulty"] <= 1)).all():
        raise ValueError("Invalid difficulty labels")
    if "counterfactual_actions" in data or "counterfactual_valid" in data:
        labels = data.get("counterfactual_actions")
        valid = data.get("counterfactual_valid")
        if (
            labels is None
            or valid is None
            or labels.shape != (count, 7)
            or valid.shape != (count, 7)
        ):
            raise ValueError("Counterfactual labels require paired [time,7] arrays")
        if (
            labels.dtype != np.int64
            or valid.dtype != np.bool_
            or not ((labels >= 0) & (labels < 13)).all()
        ):
            raise ValueError("Invalid counterfactual action labels")
    return data


def episode_tensors(data: dict[str, np.ndarray], *, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Full episode retains recurrent context across every command switch."""
    result = {
        key: torch.as_tensor(data[key], device=device)[None]
        for key in (
            "frames",
            "scalars",
            "previous_actions",
            "actions",
            "commands",
            "difficulty",
            "valid",
        )
    }
    result["frames"] = result["frames"].unsqueeze(2)
    result["difficulty"] = result["difficulty"].unsqueeze(-1)
    result["masks"] = 1 - torch.as_tensor(data["episode_start"], device=device).float()[None]
    return result
