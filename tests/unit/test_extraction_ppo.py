from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from botcolosseo.training.extraction_ppo import TeacherAnchoredPPOTrainer
from botcolosseo.training.ppo import PPOBatch


class TinyActorCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        frames,
        scalars,
        previous_actions,
        masks,
        privileged,
        hidden,
    ):
        del scalars, previous_actions, masks, privileged
        batch, time = frames.shape[:2]
        return SimpleNamespace(
            logits=self.logits.expand(batch, time, 2),
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
