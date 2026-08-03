from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from botcolosseo.training.extraction_ppo import (
    TeacherAnchoredPPOTrainer,
    main_learning_rate_at_step,
    teacher_coefficient_at_step,
    visual_learning_rate_at_step,
)
from botcolosseo.training.ppo import PPOBatch


class TinyActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([2.0, -1.0]))

    def forward(
        self,
        frames,
        scalars,
        previous_actions,
        masks,
        hidden=None,
    ):
        del scalars, previous_actions, masks
        batch, time = frames.shape[:2]
        if hidden is None:
            hidden = torch.zeros(1, batch, 1, device=frames.device)
        return SimpleNamespace(
            logits=self.logits.expand(batch, time, 2),
            hidden=hidden,
        )


class TinyActorCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = TinyActor()
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    @property
    def logits(self):
        return self.actor.logits

    def forward(
        self,
        frames,
        scalars,
        previous_actions,
        masks,
        privileged,
        hidden,
    ):
        del privileged
        batch, time = frames.shape[:2]
        actor = self.actor(
            frames,
            scalars,
            previous_actions,
            masks,
            hidden,
        )
        return SimpleNamespace(
            logits=actor.logits,
            values=self.value.expand(batch, time),
            hidden=hidden,
        )


def batch(model: TinyActorCritic) -> PPOBatch:
    frames = torch.zeros(1, 2, 1, 84, 84, dtype=torch.uint8)
    with torch.no_grad():
        output = model(
            frames,
            torch.zeros(1, 2, 1),
            torch.zeros(1, 2, dtype=torch.long),
            torch.tensor([[0.0, 1.0]]),
            torch.zeros(1, 2, 1),
            torch.zeros(1, 1, 1),
        )
        distribution = torch.distributions.Categorical(logits=output.logits)
        actions = output.logits.argmax(dim=-1)
    return PPOBatch(
        frames=frames,
        scalars=torch.zeros(1, 2, 1),
        previous_actions=torch.zeros(1, 2, dtype=torch.long),
        masks=torch.tensor([[0.0, 1.0]]),
        privileged=torch.zeros(1, 2, 1),
        initial_hidden=torch.zeros(1, 1, 1),
        actions=actions,
        old_log_probs=distribution.log_prob(actions),
        old_values=output.values,
        advantages=torch.zeros(1, 2),
        returns=output.values,
        loss_mask=torch.ones(1, 2, dtype=torch.bool),
        teacher_actions=torch.ones(1, 2, dtype=torch.long),
        teacher_mask=torch.tensor([[True, False]]),
    )


def replay_batch() -> dict[str, torch.Tensor]:
    return {
        "frames": torch.zeros(1, 2, 1, 84, 84, dtype=torch.uint8),
        "scalars": torch.zeros(1, 2, 1),
        "previous_actions": torch.zeros(1, 2, dtype=torch.long),
        "masks": torch.tensor([[0.0, 1.0]]),
        "actions": torch.tensor([[0, 1]]),
        "valid": torch.tensor([[True, False]]),
    }


def test_teacher_anchor_adds_masked_cross_entropy() -> None:
    model = TinyActorCritic()
    trainer = TeacherAnchoredPPOTrainer.create(
        model,
        teacher_coefficient=0.5,
        learning_rate=1e-3,
        total_updates=2,
        gradient_clip=1,
        policy_clip=0.2,
        value_clip=0.2,
        value_coefficient=0,
        entropy_coefficient=0,
        max_kl=1,
    )

    metrics = trainer.evaluate(batch(model))

    assert trainer.last_teacher_loss == pytest.approx(
        torch.nn.functional.cross_entropy(
            torch.tensor([[2.0, -1.0]]),
            torch.tensor([1]),
        ).item()
    )
    assert trainer.last_teacher_agreement == 0
    assert trainer.last_supervised_tokens == 1
    assert metrics.total_loss == pytest.approx(0.5 * trainer.last_teacher_loss)


def test_teacher_anchor_requires_supervision() -> None:
    model = TinyActorCritic()
    trainer = TeacherAnchoredPPOTrainer.create(
        model,
        teacher_coefficient=0.1,
        learning_rate=1e-3,
        total_updates=2,
        gradient_clip=1,
        policy_clip=0.2,
        value_clip=0.2,
        value_coefficient=0,
        entropy_coefficient=0,
        max_kl=1,
    )
    unsupervised = batch(model)
    unsupervised = PPOBatch(
        **{
            **unsupervised.__dict__,
            "teacher_actions": None,
            "teacher_mask": None,
        }
    )

    with pytest.raises(ValueError, match="missing supervision"):
        trainer.evaluate(unsupervised)


def test_environment_step_schedules_match_curriculum_boundaries() -> None:
    assert teacher_coefficient_at_step(100_000) == pytest.approx(1.0)
    assert teacher_coefficient_at_step(350_000) == pytest.approx(0.6)
    assert teacher_coefficient_at_step(600_000) == pytest.approx(0.2)
    assert main_learning_rate_at_step(
        0, total_steps=1_000_000, initial_rate=1e-5, final_rate=1e-6
    ) == pytest.approx(1e-5)
    assert main_learning_rate_at_step(
        1_000_000,
        total_steps=1_000_000,
        initial_rate=1e-5,
        final_rate=1e-6,
    ) == pytest.approx(1e-6)
    assert visual_learning_rate_at_step(600_000) == 0
    assert visual_learning_rate_at_step(610_000) == pytest.approx(2.5e-7)
    assert visual_learning_rate_at_step(620_000) == pytest.approx(5e-7)
    assert visual_learning_rate_at_step(1_000_000) == pytest.approx(1e-7)


def test_visual_curriculum_optimizer_updates_from_environment_steps() -> None:
    model = TinyActorCritic()
    trainer = TeacherAnchoredPPOTrainer.create(
        model,
        teacher_coefficient=1.0,
        learning_rate=1e-5,
        final_learning_rate=1e-6,
        visual_parameters=(model.logits,),
        total_environment_steps=1_000_000,
        total_updates=2,
        gradient_clip=1,
        policy_clip=0.2,
        value_clip=0.2,
        value_coefficient=0,
        entropy_coefficient=0,
        max_kl=1,
    )

    trainer.set_environment_steps(610_000)

    assert trainer.teacher_coefficient == pytest.approx(0.2)
    assert trainer.optimizer.param_groups[0]["lr"] < 1e-5
    assert trainer.optimizer.param_groups[1]["lr"] == pytest.approx(2.5e-7)


def test_conservative_anchors_preserve_frozen_reference_and_mask_replay() -> None:
    model = TinyActorCritic()
    reference = copy.deepcopy(model.actor)
    trainer = TeacherAnchoredPPOTrainer.create(
        model,
        teacher_coefficient=0.1,
        reference_actor=reference,
        reference_kl_coefficient=1.0,
        replay_coefficient=1.0,
        learning_rate=1e-3,
        total_updates=2,
        gradient_clip=1,
        policy_clip=0.2,
        value_clip=0.2,
        value_coefficient=0,
        entropy_coefficient=0,
        max_kl=1,
    )

    trainer.evaluate(batch(model), replay_batch())

    assert trainer.last_reference_kl == pytest.approx(0, abs=1e-7)
    assert trainer.last_replay_loss == pytest.approx(
        torch.nn.functional.cross_entropy(
            torch.tensor([[2.0, -1.0]]),
            torch.tensor([0]),
        ).item()
    )
    assert trainer.last_replay_agreement == 1
    assert trainer.last_replay_tokens == 1
    assert all(not parameter.requires_grad for parameter in reference.parameters())
    optimized = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(id(parameter) not in optimized for parameter in reference.parameters())

    with torch.no_grad():
        model.logits[1].add_(1)
    trainer.evaluate(batch(model), replay_batch())

    assert trainer.last_reference_kl > 0
    trainer.train_step(batch(model), replay_batch())
    assert all(parameter.grad is None for parameter in reference.parameters())


def test_conservative_replay_is_required_when_enabled() -> None:
    model = TinyActorCritic()
    trainer = TeacherAnchoredPPOTrainer.create(
        model,
        teacher_coefficient=0.1,
        replay_coefficient=1.0,
        learning_rate=1e-3,
        total_updates=2,
        gradient_clip=1,
        policy_clip=0.2,
        value_clip=0.2,
        value_coefficient=0,
        entropy_coefficient=0,
        max_kl=1,
    )

    with pytest.raises(ValueError, match="replay batch is missing"):
        trainer.evaluate(batch(model))
