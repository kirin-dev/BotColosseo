from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from botcolosseo.agents.extraction_model import (
    EXTRACTION_PRIVILEGED_DIM,
    ExtractionResidualActor,
    ExtractionResidualStyleActorCritic,
    configure_extraction_actor_for_visual_curriculum,
    create_extraction_actor,
    create_extraction_actor_critic,
    freeze_extraction_actor_backbone,
)
from botcolosseo.agents.extraction_teachers import ExtractionStyle
from botcolosseo.data.extraction_demonstrations import (
    ExtractionDemonstrationBuffer,
    write_extraction_shard,
)
from botcolosseo.envs.extraction_types import ExtractionActorObservation
from botcolosseo.training.extraction_bc import (
    ExtractionChunkDataset,
    extraction_manifest_teacher_sha256,
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


def test_extraction_manifest_teacher_identity_is_validated(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"teacher_implementation_sha256": "a" * 64}),
        encoding="utf-8",
    )

    assert extraction_manifest_teacher_sha256(manifest) == "a" * 64
    manifest.write_text(
        json.dumps({"teacher_implementation_sha256": "not-a-hash"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Teacher SHA-256"):
        extraction_manifest_teacher_sha256(manifest)


def test_style_actor_freezes_base_and_trains_small_branch() -> None:
    model = ExtractionResidualActor(create_extraction_actor())

    assert all(not parameter.requires_grad for parameter in model.base.parameters())
    assert any(parameter.requires_grad for parameter in model.adapter.parameters())
    assert any(parameter.requires_grad for parameter in model.policy.parameters())
    assert sum(parameter.numel() for parameter in model.trainable_parameters()) < 30_000
    assert isinstance(model.initial_state(1, device="cpu"), torch.Tensor)


def test_post_cache_dataset_masks_pre_conversion_actions(tmp_path: Path) -> None:
    buffer = ExtractionDemonstrationBuffer()
    for index, carried in enumerate((10, 10, 85, 85)):
        item = observation()
        item = ExtractionActorObservation(
            **{
                **item.__dict__,
                "carried_value": carried,
                "free_slots": 2 if carried == 10 else 0,
                "minimum_slot_value": 10,
            }
        )
        buffer.append(
            item,
            teacher_action=1,
            episode_start=index == 0,
            style=ExtractionStyle.AGGRESSIVE,
            train_seed=7,
        )
    shard = write_extraction_shard(buffer.arrays(), tmp_path / "train.npz")
    dataset = ExtractionChunkDataset(
        (shard,),
        chunk_length=4,
        supervision_mode="post-cache",
    )

    assert dataset[0]["valid"].tolist() == [False, False, True, True]


def test_extraction_actor_critic_keeps_privileged_state_out_of_public_actor() -> None:
    model = create_extraction_actor_critic()
    frames = torch.zeros(2, 3, 1, 84, 84, dtype=torch.uint8)
    scalars = torch.zeros(2, 3, 9)
    previous = torch.zeros(2, 3, dtype=torch.long)
    masks = torch.ones(2, 3)
    privileged = torch.zeros(2, 3, EXTRACTION_PRIVILEGED_DIM)

    output = model(frames, scalars, previous, masks, privileged)
    public = model.actor(frames, scalars, previous, masks)

    assert output.logits.shape == (2, 3, 13)
    assert output.values.shape == (2, 3)
    assert torch.equal(output.logits, public.logits)
    assert model.initial_state(2, device="cpu").shape == (1, 2, 256)


def test_extraction_value_loss_does_not_rewrite_bc_actor() -> None:
    model = create_extraction_actor_critic()
    frames = torch.zeros(1, 2, 1, 84, 84, dtype=torch.uint8)
    scalars = torch.zeros(1, 2, 9)
    previous = torch.zeros(1, 2, dtype=torch.long)
    masks = torch.ones(1, 2)
    privileged = torch.zeros(1, 2, EXTRACTION_PRIVILEGED_DIM)

    output = model(frames, scalars, previous, masks, privileged)
    output.values.sum().backward()

    assert all(parameter.grad is None for parameter in model.actor.parameters())
    assert any(
        parameter.grad is not None
        for parameter in model.privileged_encoder.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.value.parameters())


def test_strong_calibration_freezes_only_actor_backbone() -> None:
    model = create_extraction_actor_critic()

    freeze_extraction_actor_backbone(model)

    assert all(
        not parameter.requires_grad
        for parameter in model.actor.visual_encoder.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.actor.scalar_encoder.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.actor.recurrent.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.actor.policy.parameters())
    assert all(
        parameter.requires_grad
        for parameter in model.privileged_encoder.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.value.parameters())


def test_visual_curriculum_trains_only_final_convolution_from_visual_path() -> None:
    model = create_extraction_actor_critic()

    visual_parameters = configure_extraction_actor_for_visual_curriculum(model)

    assert visual_parameters == tuple(model.actor.visual_encoder[4].parameters())
    assert all(parameter.requires_grad for parameter in visual_parameters)
    assert all(
        not parameter.requires_grad
        for index, module in enumerate(model.actor.visual_encoder)
        if index != 4
        for parameter in module.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.actor.scalar_encoder.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.actor.recurrent.parameters())
    assert all(parameter.requires_grad for parameter in model.actor.policy.parameters())


def test_extraction_style_is_zero_initialized_bounded_delta_over_frozen_base() -> None:
    model = ExtractionResidualStyleActorCritic(
        create_extraction_actor_critic(),
        bottleneck=16,
        max_delta=1.5,
    )
    frames = torch.zeros(1, 2, 1, 84, 84, dtype=torch.uint8)
    scalars = torch.zeros(1, 2, 9)
    previous = torch.zeros(1, 2, dtype=torch.long)
    masks = torch.ones(1, 2)
    privileged = torch.zeros(1, 2, EXTRACTION_PRIVILEGED_DIM)

    output = model(frames, scalars, previous, masks, privileged)
    public = model.public_actor()(frames, scalars, previous, masks)

    assert torch.equal(output.logits, output.base_logits)
    assert torch.equal(output.logits, public.logits)
    assert all(
        not parameter.requires_grad for parameter in model.base.parameters()
    )
    assert any(parameter.requires_grad for parameter in model.delta_policy.parameters())
    assert any(
        parameter.requires_grad
        for parameter in model.style_privileged_encoder.parameters()
    )
    assert any(parameter.requires_grad for parameter in model.style_value.parameters())
    assert model.initial_state(1, device="cpu").shape == (1, 1, 256)
