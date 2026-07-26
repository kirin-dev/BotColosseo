from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from botcolosseo.data.extraction_demonstrations import (
    EXTRACTION_SCALAR_DIM,
    extraction_scalars,
    load_extraction_shard,
)
from botcolosseo.envs.extraction_types import ExtractionActorObservation


class ExtractionChunkDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        shard_paths: tuple[Path, ...],
        *,
        chunk_length: int,
        max_transitions: int | None = None,
        supervision_mode: str = "all",
    ) -> None:
        if not shard_paths or chunk_length <= 0:
            raise ValueError("Extraction dataset inputs must be nonempty")
        if max_transitions is not None and max_transitions <= 0:
            raise ValueError("Extraction max transitions must be positive")
        if supervision_mode not in {"all", "post-cache"}:
            raise ValueError("Unsupported Extraction supervision mode")
        loaded: dict[str, list[np.ndarray]] = {}
        remaining = max_transitions
        for path in shard_paths:
            arrays = load_extraction_shard(path)
            take = (
                len(arrays["frame"])
                if remaining is None
                else min(remaining, len(arrays["frame"]))
            )
            for name, array in arrays.items():
                loaded.setdefault(name, []).append(array[:take])
            if remaining is not None:
                remaining -= take
                if remaining == 0:
                    break
        self._arrays = {
            name: np.concatenate(parts, axis=0) for name, parts in loaded.items()
        }
        self.chunk_length = chunk_length
        self.transition_count = len(self._arrays["frame"])
        self._supervised = np.asarray(
            self._arrays["valid_mask"],
            dtype=np.bool_,
        )
        if supervision_mode == "post-cache":
            self._supervised &= self._arrays["scalars"][:, 2] > 10 / 150
        self._starts = tuple(
            start
            for start in range(0, self.transition_count, chunk_length)
            if bool(np.any(self._supervised[start : start + chunk_length]))
        )
        if not self._starts:
            raise ValueError("Extraction dataset contains no chunks")

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = self._starts[index]
        stop = min(start + self.chunk_length, self.transition_count)
        size = stop - start
        frames = np.zeros((self.chunk_length, 1, 84, 84), dtype=np.uint8)
        scalars = np.zeros(
            (self.chunk_length, EXTRACTION_SCALAR_DIM),
            dtype=np.float32,
        )
        previous_actions = np.zeros(self.chunk_length, dtype=np.int64)
        actions = np.zeros(self.chunk_length, dtype=np.int64)
        valid = np.zeros(self.chunk_length, dtype=np.bool_)
        masks = np.zeros(self.chunk_length, dtype=np.float32)
        frames[:size, 0] = self._arrays["frame"][start:stop]
        scalars[:size] = self._arrays["scalars"][start:stop]
        previous_actions[:size] = self._arrays["previous_action"][start:stop]
        actions[:size] = self._arrays["teacher_action"][start:stop]
        valid[:size] = self._supervised[start:stop]
        episode_start = self._arrays["episode_start"][start:stop]
        masks[:size] = (~episode_start).astype(np.float32)
        masks[0] = 0.0
        return {
            "frames": torch.from_numpy(frames),
            "scalars": torch.from_numpy(scalars),
            "previous_actions": torch.from_numpy(previous_actions),
            "actions": torch.from_numpy(actions),
            "masks": torch.from_numpy(masks),
            "valid": torch.from_numpy(valid),
        }


def load_extraction_shard_paths(manifest_path: Path) -> tuple[Path, ...]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("split") not in {"train", "validation"}:
        raise ValueError("Extraction BC may only load train or validation data")
    if payload.get("test_cases_accessed") is not False:
        raise ValueError("Extraction BC manifest must deny test-case access")
    paths = tuple(manifest_path.parent / item["file"] for item in payload["shards"])
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("Extraction BC manifest references missing shards")
    return paths


def extraction_observation_tensors(
    observation: ExtractionActorObservation,
    *,
    episode_start: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    frame = torch.from_numpy(np.array(observation.frame, copy=True)).to(device)
    frame = frame.reshape(1, 1, 1, 84, 84)
    scalars = torch.from_numpy(extraction_scalars(observation)).to(device)
    scalars = scalars.reshape(1, 1, EXTRACTION_SCALAR_DIM)
    previous_action = torch.tensor(
        [[observation.previous_action]],
        dtype=torch.long,
        device=device,
    )
    mask = torch.tensor(
        [[0.0 if episode_start else 1.0]],
        dtype=torch.float32,
        device=device,
    )
    return frame, scalars, previous_action, mask
