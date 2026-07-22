from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DEMONSTRATION_FIELDS = (
    "frame",
    "scalars",
    "previous_action",
    "teacher_action",
    "episode_start",
    "valid_mask",
    "opponent_id",
    "task_id",
    "train_seed",
)

PRIVILEGED_KEYS = frozenset(
    {
        "player_x",
        "player_y",
        "host_x",
        "host_y",
        "host_angle",
        "host_region",
        "opponent_x",
        "opponent_y",
        "opponent_angle",
        "opponent_region",
        "core_x",
        "core_y",
        "carrier",
        "engine_tic",
        "privileged_state",
    }
)


def find_privileged_keys(value: Any) -> tuple[str, ...]:
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                key_text = str(key)
                if key_text in PRIVILEGED_KEYS:
                    found.add(key_text)
                visit(nested)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for nested in item:
                visit(nested)

    visit(value)
    return tuple(sorted(found))


def validate_demonstration_shard(arrays: Mapping[str, np.ndarray]) -> int:
    if set(arrays) != set(DEMONSTRATION_FIELDS):
        raise ValueError("Demonstration shard fields do not match the frozen schema")
    lengths = {np.asarray(value).shape[0] for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("Demonstration fields have inconsistent lengths")
    size = lengths.pop()
    if size <= 0:
        raise ValueError("Demonstration shard must not be empty")
    expected = {
        "frame": ((size, 84, 84), np.dtype(np.uint8)),
        "scalars": ((size, 6), np.dtype(np.float32)),
        "previous_action": ((size,), np.dtype(np.int8)),
        "teacher_action": ((size,), np.dtype(np.int8)),
        "episode_start": ((size,), np.dtype(np.bool_)),
        "valid_mask": ((size,), np.dtype(np.bool_)),
        "opponent_id": ((size,), np.dtype(np.uint8)),
        "task_id": ((size,), np.dtype(np.uint8)),
        "train_seed": ((size,), np.dtype(np.int64)),
    }
    for name, (shape, dtype) in expected.items():
        array = np.asarray(arrays[name])
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(
                f"Invalid {name} shape/dtype: {array.shape}/{array.dtype}"
            )
    if not bool(arrays["episode_start"][0]):
        raise ValueError("Every shard must begin at an episode boundary")
    if not bool(np.all(arrays["valid_mask"])):
        raise ValueError("Stored demonstrations must have an all-true valid mask")
    if np.any(arrays["previous_action"] < 0) or np.any(
        arrays["previous_action"] >= 13
    ):
        raise ValueError("Previous action ID is outside the fixed action space")
    if np.any(arrays["teacher_action"] < 0) or np.any(
        arrays["teacher_action"] >= 13
    ):
        raise ValueError("Teacher action ID is outside the fixed action space")
    if np.any(arrays["opponent_id"] >= 5):
        raise ValueError("Opponent ID is outside the frozen opponent set")
    if np.any(arrays["task_id"] >= 6):
        raise ValueError("Task ID is outside the frozen Teacher mode set")
    scalars = arrays["scalars"]
    if not np.isfinite(scalars).all() or np.any(scalars < 0) or np.any(scalars > 1):
        raise ValueError("Actor scalars must be finite and normalized to [0, 1]")
    return size
