from __future__ import annotations

import torch

from botcolosseo.training.gae import normalize_advantages
from botcolosseo.training.ppo import (
    PPOBatch,
    PPOLoss,
    PPOTrainer,
    ppo_loss,
)


class TeacherAnchoredPPOTrainer(PPOTrainer):
    """PPO with an on-policy Strong Teacher anchor."""

    def __init__(self, *args, teacher_coefficient: float, **kwargs) -> None:
        if teacher_coefficient <= 0:
            raise ValueError("Teacher auxiliary coefficient must be positive")
        super().__init__(*args, **kwargs)
        self.teacher_coefficient = teacher_coefficient
        self.last_teacher_loss = 0.0
        self.last_teacher_agreement = 0.0
        self.last_supervised_tokens = 0

    @classmethod
    def create(
        cls,
        model: torch.nn.Module,
        *,
        teacher_coefficient: float,
        learning_rate: float,
        total_updates: int,
        gradient_clip: float,
        policy_clip: float,
        value_clip: float,
        value_coefficient: float,
        entropy_coefficient: float,
        max_kl: float,
        weight_decay: float = 0.0,
    ) -> TeacherAnchoredPPOTrainer:
        if learning_rate <= 0 or total_updates <= 0 or weight_decay < 0:
            raise ValueError("Invalid teacher-anchored PPO optimizer settings")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_updates,
        )
        return cls(
            model,
            optimizer,
            scheduler,
            teacher_coefficient=teacher_coefficient,
            gradient_clip=gradient_clip,
            policy_clip=policy_clip,
            value_clip=value_clip,
            value_coefficient=value_coefficient,
            entropy_coefficient=entropy_coefficient,
            max_kl=max_kl,
        )

    def _loss(self, batch: PPOBatch) -> PPOLoss:
        if batch.teacher_actions is None or batch.teacher_mask is None:
            raise ValueError("Teacher-anchored PPO batch is missing supervision")
        supervised = batch.teacher_mask & batch.loss_mask
        if not bool(supervised.any()):
            raise ValueError("Teacher-anchored PPO batch has no supervised tokens")
        output = self.model(
            batch.frames,
            batch.scalars,
            batch.previous_actions,
            batch.masks,
            batch.privileged,
            batch.initial_hidden,
        )
        loss = ppo_loss(
            logits=output.logits,
            values=output.values,
            actions=batch.actions,
            old_log_probs=batch.old_log_probs,
            old_values=batch.old_values,
            advantages=normalize_advantages(
                batch.advantages,
                batch.loss_mask,
            ),
            returns=batch.returns,
            valid=batch.loss_mask,
            policy_clip=self.policy_clip,
            value_clip=self.value_clip,
            value_coefficient=self.value_coefficient,
            entropy_coefficient=self.entropy_coefficient,
            max_kl=self.max_kl,
        )
        logits = output.logits[supervised]
        actions = batch.teacher_actions[supervised]
        teacher_loss = torch.nn.functional.cross_entropy(logits, actions)
        if not bool(torch.isfinite(teacher_loss)):
            raise FloatingPointError("Teacher auxiliary loss is not finite")
        self.last_teacher_loss = float(teacher_loss.detach())
        self.last_teacher_agreement = float(
            (logits.argmax(dim=-1) == actions).float().mean().detach()
        )
        self.last_supervised_tokens = int(supervised.sum())
        return loss._replace(
            total_loss=(
                loss.total_loss + self.teacher_coefficient * teacher_loss
            )
        )
