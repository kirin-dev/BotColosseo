from __future__ import annotations

import torch

from botcolosseo.agents.style_model import RoutedStyledActorCritic, StyledActorCritic
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


def masked_teacher_cross_entropy(
    logits: torch.Tensor,
    teacher_actions: torch.Tensor,
    supervised: torch.Tensor,
) -> tuple[torch.Tensor, float, int]:
    if logits.shape[:-1] != teacher_actions.shape or supervised.shape != teacher_actions.shape:
        raise ValueError("Teacher tensors must match the policy token shape")
    if teacher_actions.dtype != torch.long or supervised.dtype is not torch.bool:
        raise ValueError("Teacher actions and mask have invalid dtypes")
    if not bool(supervised.any()):
        raise ValueError("Teacher mask must select at least one item")
    selected_logits = logits[supervised]
    selected_actions = teacher_actions[supervised]
    if int(selected_actions.min()) < 0 or int(selected_actions.max()) >= logits.shape[-1]:
        raise ValueError("Teacher action is outside the policy action space")
    loss = torch.nn.functional.cross_entropy(selected_logits, selected_actions)
    agreement = float(
        (selected_logits.argmax(dim=-1) == selected_actions).float().mean().detach()
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Teacher auxiliary loss is not finite")
    return loss, agreement, int(supervised.sum())


class StylePPOTrainer(PPOTrainer):
    def __init__(self, *args, beta_kl: float, eta_aux: float = 0.0, **kwargs) -> None:
        if beta_kl < 0 or eta_aux < 0:
            raise ValueError("Style loss coefficients must be nonnegative")
        super().__init__(*args, **kwargs)
        self.beta_kl = beta_kl
        self.eta_aux = eta_aux
        self.last_style_kl = 0.0
        self.last_auxiliary_loss = 0.0
        self.last_teacher_agreement = 0.0
        self.last_supervised_tokens = 0

    @classmethod
    def create(
        cls,
        model: StyledActorCritic | RoutedStyledActorCritic,
        *,
        beta_kl: float,
        eta_aux: float = 0.0,
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
            eta_aux=eta_aux,
            gradient_clip=gradient_clip,
            policy_clip=policy_clip,
            value_clip=value_clip,
            value_coefficient=value_coefficient,
            entropy_coefficient=entropy_coefficient,
            max_kl=max_kl,
        )

    def _loss(self, batch: PPOBatch) -> PPOLoss:
        normalized = normalize_advantages(batch.advantages, batch.loss_mask)
        model_kwargs = {}
        if isinstance(self.model, RoutedStyledActorCritic):
            if batch.route_modes is None:
                raise ValueError("Explorer PPO batch is missing route modes")
            model_kwargs["route_modes"] = batch.route_modes
        output = self.model(
            batch.frames,
            batch.scalars,
            batch.previous_actions,
            batch.masks,
            batch.privileged,
            batch.initial_hidden,
            **model_kwargs,
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
        auxiliary = output.logits.sum() * 0.0
        self.last_auxiliary_loss = 0.0
        self.last_teacher_agreement = 0.0
        self.last_supervised_tokens = 0
        if self.eta_aux > 0:
            if batch.teacher_actions is None or batch.teacher_mask is None:
                raise ValueError("Auxiliary PPO batch is missing Teacher supervision")
            supervised = batch.teacher_mask & batch.loss_mask
            if bool(supervised.any()):
                auxiliary, agreement, count = masked_teacher_cross_entropy(
                    output.logits,
                    batch.teacher_actions,
                    supervised,
                )
                self.last_auxiliary_loss = float(auxiliary.detach())
                self.last_teacher_agreement = agreement
                self.last_supervised_tokens = count
        return loss._replace(
            total_loss=(
                loss.total_loss
                + self.beta_kl * style_kl
                + self.eta_aux * auxiliary
            )
        )
