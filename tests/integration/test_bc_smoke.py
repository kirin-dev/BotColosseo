from pathlib import Path

import pytest
import torch

from botcolosseo.agents.model import RecurrentActor
from botcolosseo.training.bc import (
    BCTrainer,
    DemonstrationChunkDataset,
    DeterministicBatchStream,
)


@pytest.mark.integration
@pytest.mark.timeout(90)
def test_real_demonstration_batch_overfits_in_50_updates() -> None:
    root = Path(__file__).resolve().parents[2]
    shard = root / "data/generated/m2/train/train-00000.npz"
    if not shard.is_file():
        pytest.skip("Full M2 demonstrations are not generated")
    data = DemonstrationChunkDataset(
        (shard,), chunk_length=8, max_transitions=64
    )
    batch = DeterministicBatchStream(data, batch_size=8, seed=7).batch(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = BCTrainer.create(
        RecurrentActor().to(device),
        learning_rate=1e-3,
        gradient_clip=1.0,
        total_updates=50,
    )

    initial = trainer.evaluate_batch(batch)
    for _ in range(50):
        trainer.train_step(batch)
    final = trainer.evaluate_batch(batch)

    assert final.loss < initial.loss * 0.5
    assert final.accuracy > initial.accuracy
