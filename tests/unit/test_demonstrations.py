from pathlib import Path

import numpy as np
import pytest

from botcolosseo.data.demonstrations import (
    DemonstrationBuffer,
    load_demonstration_shard,
    sha256_file,
    write_demonstration_shard,
)
from botcolosseo.envs.duel_types import DuelActorObservation


def observation(previous_action: int = 0) -> DuelActorObservation:
    return DuelActorObservation(
        frame=np.arange(84 * 84, dtype=np.uint8).reshape(84, 84),
        health=100.0,
        armor=50.0,
        ammo=25.0,
        own_score=1,
        opponent_score=2,
        has_core=True,
        previous_action=previous_action,
    )


def test_bounded_buffer_normalizes_public_observation_and_preserves_boundaries() -> None:
    buffer = DemonstrationBuffer(capacity=2)
    buffer.append(
        observation(),
        teacher_action=3,
        episode_start=True,
        opponent_id=4,
        task_id=2,
        train_seed=17,
    )
    buffer.append(
        observation(3),
        teacher_action=4,
        episode_start=False,
        opponent_id=4,
        task_id=3,
        train_seed=17,
    )

    arrays = buffer.arrays()
    assert arrays["frame"].shape == (2, 84, 84)
    np.testing.assert_allclose(
        arrays["scalars"][0], [0.5, 0.25, 0.25, 1 / 3, 2 / 3, 1.0]
    )
    assert arrays["episode_start"].tolist() == [True, False]
    with pytest.raises(BufferError):
        buffer.append(
            observation(),
            teacher_action=0,
            episode_start=False,
            opponent_id=0,
            task_id=0,
            train_seed=0,
        )


def test_npz_bytes_and_hash_are_deterministic_and_atomic(tmp_path: Path) -> None:
    buffer = DemonstrationBuffer(capacity=1)
    buffer.append(
        observation(),
        teacher_action=3,
        episode_start=True,
        opponent_id=1,
        task_id=2,
        train_seed=17,
    )
    first = write_demonstration_shard(buffer.arrays(), tmp_path / "first.npz")
    second = write_demonstration_shard(buffer.arrays(), tmp_path / "second.npz")

    assert first.read_bytes() == second.read_bytes()
    assert sha256_file(first) == sha256_file(second)
    loaded = load_demonstration_shard(first)
    assert loaded["teacher_action"].tolist() == [3]
    assert not list(tmp_path.glob("*.tmp"))
