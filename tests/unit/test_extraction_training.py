from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from botcolosseo.agents.extraction_model import (
    ExtractionResidualActor,
    create_extraction_actor,
)
from botcolosseo.agents.extraction_teachers import ExtractionStyle
from botcolosseo.data.extraction_demonstrations import (
    ExtractionDemonstrationBuffer,
    write_extraction_shard,
)
from botcolosseo.envs.extraction_types import ExtractionActorObservation
from botcolosseo.training.extraction_bc import (
    ExtractionChunkDataset,
    load_extraction_shard_paths,
)


def observation() -> ExtractionActorObservation:
    return ExtractionActorObservation(
        frame=np.zeros((84, 84), dtype=np.uint8),
        health=100,
        ammo=30,
        carried_value=0,
        free_slots=3,
        minimum_slot_value=0,
        banked_value=0,
        extraction_open=False,
        extraction_progress=0,
        remaining_time=75,
        previous_action=0,
    )


def test_extraction_dataset_and_actor_shapes(tmp_path: Path) -> None:
    buffer = ExtractionDemonstrationBuffer()
    for index in range(3):
        buffer.append(
            observation(),
            teacher_action=1,
            episode_start=index == 0,
            style=ExtractionStyle.STRONG,
            train_seed=7,
        )
    shard = write_extraction_shard(buffer.arrays(), tmp_path / "train-00000.npz")
    manifest = tmp_path / "train-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "split": "train",
                "test_cases_accessed": False,
                "shards": [{"file": shard.name}],
            }
        ),
        encoding="utf-8",
    )
    dataset = ExtractionChunkDataset(
        load_extraction_shard_paths(manifest),
        chunk_length=4,
    )
    batch = dataset[0]
    model = create_extraction_actor()
    output = model(
        batch["frames"].unsqueeze(0),
        batch["scalars"].unsqueeze(0),
        batch["previous_actions"].unsqueeze(0),
        batch["masks"].unsqueeze(0),
    )

    assert output.logits.shape == (1, 4, 13)
    assert batch["valid"].tolist() == [True, True, True, False]


def test_style_actor_freezes_base_and_trains_small_branch() -> None:
    model = ExtractionResidualActor(create_extraction_actor())

    assert all(not parameter.requires_grad for parameter in model.base.parameters())
    assert any(parameter.requires_grad for parameter in model.adapter.parameters())
    assert any(parameter.requires_grad for parameter in model.policy.parameters())
    assert sum(parameter.numel() for parameter in model.trainable_parameters()) < 30_000
    assert isinstance(model.initial_state(1, device="cpu"), torch.Tensor)
