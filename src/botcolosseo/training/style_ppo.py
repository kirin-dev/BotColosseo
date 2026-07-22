from __future__ import annotations

import torch

from botcolosseo.agents.style_model import StyledActorCritic
from botcolosseo.training.gae import normalize_advantages
from botcolosseo.training.ppo import PPOBatch, PPOLoss, PPOTrainer, ppo_loss


def categorical_style_kl(
    style_logits: torch.Tensor,
    base_logits: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Compute D_KL(style || base) on selected recurrent tokens."""
    if style_logits.shape != base_logits.shape or style_logits.ndim < 2:
        raise ValueError("Style and base logits must share a categorical shape")
    if valid.shape != style_logits.shape[:-1] or valid.dtype is not torch.bool:
        raise ValueError("Style KL mask has the wrong shape or dtype")
    if not bool(valid.any()):
        raise ValueError("Style KL mask must select at least one item")
    style_log_prob = torch.log_softmax(style_logits[valid], dim=-1)
    base_log_prob = torch.log_softmax(base_logits[valid], dim=-1)
    style_prob = style_log_prob.exp()
    divergence = (style_prob * (style_log_prob - base_log_prob)).sum(dim=-1).mean()
    if not bool(torch.isfinite(divergence)):
        raise FloatingPointError("Style KL is not finite")
    return divergence


class StylePPOTrainer(PPOTrainer):
    def __init__(self, *args, beta_kl: float, **kwargs) -> None:
        if beta_kl < 0:
            raise ValueError("Style KL coefficient must be nonnegative")
        super().__init__(*args, **kwargs)
        self.beta_kl = beta_kl
        self.last_style_kl = 0.0

    @classmethod
    def create(
        cls,
        model: StyledActorCritic,
        *,
        beta_kl: float,
        learning_rate: float,
        total_updates: int,
        gradient_clip: float,
        policy_clip: float,
        value_clip: float,
        value_coefficient: float,
        entropy_coefficient: float,
        max_kl: float,
        weight_decay: float = 0.0,
    ) -> StylePPOTrainer:
        if learning_rate <= 0 or total_updates <= 0 or weight_decay < 0:
            raise ValueError("Invalid style PPO optimizer settings")
        optimizer = torch.optim.AdamW(
            model.trainable_parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_updates
        )
        return cls(
            model,
            optimizer,
            scheduler,
            beta_kl=beta_kl,
            gradient_clip=gradient_clip,
            policy_clip=policy_clip,
            value_clip=value_clip,
            value_coefficient=value_coefficient,
            entropy_coefficient=entropy_coefficient,
            max_kl=max_kl,
        )

    def _loss(self, batch: PPOBatch) -> PPOLoss:
        normalized = normalize_advantages(batch.advantages, batch.loss_mask)
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
            advantages=normalized,
            returns=batch.returns,
            valid=batch.loss_mask,
            policy_clip=self.policy_clip,
            value_clip=self.value_clip,
            value_coefficient=self.value_coefficient,
            entropy_coefficient=self.entropy_coefficient,
            max_kl=self.max_kl,
        )
        style_kl = categorical_style_kl(
            output.logits, output.base_logits, batch.loss_mask
        )
        self.last_style_kl = float(style_kl.detach())
        return loss._replace(total_loss=loss.total_loss + self.beta_kl * style_kl)
