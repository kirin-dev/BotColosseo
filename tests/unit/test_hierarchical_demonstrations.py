import numpy as np
import pytest

from botcolosseo.data.hierarchical_demonstrations import load_command_episode
from botcolosseo.training.hierarchical_protocol import COMMAND_SCHEMA


def test_loader_rejects_unrecorded_command_switch(tmp_path):
    data = dict(
        frames=np.zeros((2, 84, 84), dtype=np.uint8),
        scalars=np.zeros((2, 9), dtype=np.float32),
        actions=np.zeros(2, dtype=np.int64),
        previous_actions=np.zeros(2, dtype=np.int64),
        commands=np.array([0, 1], dtype=np.int64),
        valid=np.ones(2, dtype=bool),
        episode_start=np.array([True, False]),
        command_start=np.array([True, False]),
        difficulty=np.ones(2, dtype=np.float32),
        command_schema=np.asarray(COMMAND_SCHEMA),
    )
    path = tmp_path / "episode.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError, match="boundary"):
        load_command_episode(path)
    data["command_start"][:] = True
    np.savez(path, **data)
    assert len(load_command_episode(path)["actions"]) == 2
