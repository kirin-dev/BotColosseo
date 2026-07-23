import json
from pathlib import Path

import numpy as np
import torch

from botcolosseo.agents.model import RecurrentActor
from botcolosseo.data.demonstrations import write_demonstration_shard
from botcolosseo.training.bc import (
    BCTrainer,
    BestCheckpointTracker,
    DemonstrationChunkDataset,
    DeterministicBatchStream,
    behavior_cloning_metrics,
    load_shard_paths,
    make_validation_loader,
)


def arrays(size: int = 5) -> dict[str, np.ndarray]:
    return {
        "frame": np.zeros((size, 84, 84), dtype=np.uint8),
        "scalars": np.zeros((size, 6), dtype=np.float32),
        "previous_action": np.arange(size, dtype=np.int8) % 13,
        "teacher_action": np.arange(size, dtype=np.int8) % 13,
        "episode_start": np.array([True, False, True, False, False], dtype=np.bool_)[:size],
        "valid_mask": np.ones(size, dtype=np.bool_),
        "opponent_id": np.zeros(size, dtype=np.uint8),
        "task_id": np.zeros(size, dtype=np.uint8),
        "train_seed": np.full(size, 7, dtype=np.int64),
    }


def dataset(tmp_path: Path, *, size: int = 5, chunk: int = 4):
    path = write_demonstration_shard(arrays(size), tmp_path / "data.npz")
    return DemonstrationChunkDataset((path,), chunk_length=chunk)


def test_recurrent_chunks_preserve_boundaries_and_pad_valid_mask(tmp_path: Path) -> None:
    data = dataset(tmp_path)

    first = data[0]
    second = data[1]

    assert first["frames"].shape == (4, 1, 84, 84)
    assert first["masks"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert second["masks"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert second["valid"].tolist() == [True, False, False, False]
    assert second["present"].tolist() == [True, False, False, False]
    assert first["task_ids"].tolist() == [0, 0, 0, 0]
    assert not ({"host_x", "core_x", "carrier"} & set(first))


def test_masked_cross_entropy_and_accuracy_match_hand_calculation() -> None:
    logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]]])
    actions = torch.tensor([[0, 1, 1]])
    valid = torch.tensor([[True, True, False]])

    metrics = behavior_cloning_metrics(logits, actions, valid)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :2].reshape(-1, 2), actions[:, :2].reshape(-1)
    )

    torch.testing.assert_close(metrics.loss, expected)
    assert metrics.accuracy == 1.0
    assert metrics.valid_count == 2


def test_deterministic_batch_stream_and_validation_loader(tmp_path: Path) -> None:
    data = dataset(tmp_path, size=5, chunk=2)
    first = DeterministicBatchStream(data, batch_size=2, seed=11)
    second = DeterministicBatchStream(data, batch_size=2, seed=11)

    for update in range(4):
        assert torch.equal(first.batch(update)["actions"], second.batch(update)["actions"])
    loader_a = list(make_validation_loader(data, batch_size=2))
    loader_b = list(make_validation_loader(data, batch_size=2))
    assert all(
        torch.equal(left["actions"], right["actions"])
        for left, right in zip(loader_a, loader_b, strict=True)
    )


def test_best_checkpoint_uses_objective_rate_then_validation_loss() -> None:
    tracker = BestCheckpointTracker()

    assert tracker.update(validation_loss=1.0, objective_rate=0.2, update=10)
    assert not tracker.update(validation_loss=0.5, objective_rate=0.1, update=20)
    assert tracker.update(validation_loss=1.2, objective_rate=0.3, update=30)
    assert tracker.update(validation_loss=1.1, objective_rate=0.3, update=40)
    assert tracker.best_update == 40


def test_bc_rejects_test_split_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "split": "test",
                "test_cases_accessed": False,
                "shards": [{"file": "data.npz"}],
            }
        ),
        encoding="utf-8",
    )

    try:
        load_shard_paths(manifest)
    except ValueError as error:
        assert "train or validation" in str(error)
    else:
        raise AssertionError("BC accepted a test split manifest")


def test_trainer_clips_gradients_validates_and_resumes_exactly(tmp_path: Path) -> None:
    torch.manual_seed(13)
    data = dataset(tmp_path, size=5, chunk=4)
    batch = DeterministicBatchStream(data, batch_size=2, seed=3).batch(0)
    trainer = BCTrainer.create(
        RecurrentActor(), learning_rate=1e-3, gradient_clip=0.05, total_updates=5
    )

    first = trainer.train_step(batch)
    validation = trainer.validate(make_validation_loader(data, batch_size=2))
    checkpoint = trainer.save(
        tmp_path / "bc.pt",
        config_hash="config",
        scenario_hash="scenario",
    )
    baseline = trainer.train_step(batch)
    baseline_state = {
        name: value.detach().clone() for name, value in trainer.model.state_dict().items()
    }
    resumed = BCTrainer.create(
        RecurrentActor(), learning_rate=1e-3, gradient_clip=0.05, total_updates=5
    )
    resumed.load(
        checkpoint,
        config_hash="config",
        scenario_hash="scenario",
        restore_rng=True,
    )
    repeated = resumed.train_step(batch)

    assert first.post_clip_grad_norm <= 0.050001
    assert validation.valid_count == 5
    assert repeated.loss == baseline.loss
    assert resumed.updates == trainer.updates
    assert all(
        torch.equal(value, baseline_state[name])
        for name, value in resumed.model.state_dict().items()
    )
