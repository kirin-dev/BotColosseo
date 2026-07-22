from pathlib import Path

import numpy as np
import pytest

from botcolosseo.data.schema import (
    DEMONSTRATION_FIELDS,
    find_privileged_keys,
    validate_demonstration_shard,
)


def valid_arrays(size: int = 3) -> dict[str, np.ndarray]:
    return {
        "frame": np.zeros((size, 84, 84), dtype=np.uint8),
        "scalars": np.zeros((size, 6), dtype=np.float32),
        "previous_action": np.zeros(size, dtype=np.int8),
        "teacher_action": np.ones(size, dtype=np.int8),
        "episode_start": np.array([True, *([False] * (size - 1))], dtype=np.bool_),
        "valid_mask": np.ones(size, dtype=np.bool_),
        "opponent_id": np.zeros(size, dtype=np.uint8),
        "task_id": np.zeros(size, dtype=np.uint8),
        "train_seed": np.full(size, 7, dtype=np.int64),
    }


def test_schema_has_exact_public_fields_and_rejects_recursive_privileged_keys() -> None:
    assert set(valid_arrays()) == set(DEMONSTRATION_FIELDS)
    leaked = {"metadata": {"player_x": 1.0, "nested": [{"carrier": 2}]}}

    assert find_privileged_keys(leaked) == ("carrier", "player_x")


def test_schema_validates_shapes_dtypes_boundaries_masks_and_ids() -> None:
    arrays = valid_arrays()
    assert validate_demonstration_shard(arrays) == 3

    for key, value in (
        ("frame", np.zeros((3, 83, 84), dtype=np.uint8)),
        ("scalars", np.zeros((3, 6), dtype=np.float64)),
        ("teacher_action", np.full(3, 13, dtype=np.int8)),
        ("valid_mask", np.array([True, False, True], dtype=np.bool_)),
        ("opponent_id", np.full(3, 5, dtype=np.uint8)),
    ):
        invalid = valid_arrays()
        invalid[key] = value
        with pytest.raises(ValueError):
            validate_demonstration_shard(invalid)

    no_boundary = valid_arrays()
    no_boundary["episode_start"][0] = False
    with pytest.raises(ValueError, match="boundary"):
        validate_demonstration_shard(no_boundary)


def test_schema_rejects_extra_fields_and_inconsistent_lengths(tmp_path: Path) -> None:
    extra = valid_arrays()
    extra["host_x"] = np.zeros(3)
    with pytest.raises(ValueError, match="fields"):
        validate_demonstration_shard(extra)

    short = valid_arrays()
    short["train_seed"] = np.zeros(2, dtype=np.int64)
    with pytest.raises(ValueError, match="length"):
        validate_demonstration_shard(short)
