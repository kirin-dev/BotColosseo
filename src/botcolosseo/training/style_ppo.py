from __future__ import annotations

import torch

from botcolosseo.agents.style_model import RoutedStyledActorCritic
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


def partitioned_style_kl(
    style_logits: torch.Tensor,
    base_logits: torch.Tensor,
    valid: torch.Tensor,
    opportunity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Split D_KL(style || base) inside and outside style opportunities."""
    if opportunity.shape != valid.shape or opportunity.dtype is not torch.bool:
        raise ValueError("Opportunity KL mask has the wrong shape or dtype")
    if valid.dtype is not torch.bool or not bool(valid.any()):
        raise ValueError("Style KL valid mask must select at least one item")
    zero = style_logits.sum() * 0.0
    inside_mask = valid & opportunity
    outside_mask = valid & ~opportunity
    inside = (
        categorical_style_kl(style_logits, base_logits, inside_mask)
        if bool(inside_mask.any())
        else zero
    )
    outside = (
        categorical_style_kl(style_logits, base_logits, outside_mask)
        if bool(outside_mask.any())
        else zero
    )
    return inside, outside, int(inside_mask.sum()), int(outside_mask.sum())


def preferred_action_set_margin_loss(
    style_logits: torch.Tensor,
    base_logits: torch.Tensor,
    preferred_actions: torch.Tensor,
    supervised: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, float, int]:
    """Raise preferred-set probability relative to the frozen Strong policy."""
    if style_logits.shape != base_logits.shape or style_logits.ndim < 2:
        raise ValueError("Style and base logits must share a categorical shape")
    if preferred_actions.shape != style_logits.shape or preferred_actions.dtype is not torch.bool:
        raise ValueError("Preferred action mask has the wrong shape or dtype")
    if supervised.shape != style_logits.shape[:-1] or supervised.dtype is not torch.bool:
        raise ValueError("Preference supervision mask has the wrong shape or dtype")
    if margin < 0:
        raise ValueError("Preference margin must be nonnegative")
    if not bool(supervised.any()):
        raise ValueError("Preference mask must select at least one item")
    selected_preferences = preferred_actions[supervised]
    if bool((selected_preferences.sum(dim=-1) == 0).any()):
        raise ValueError("Preference supervision requires a preferred action")
    style_log_probs = torch.log_softmax(style_logits[supervised], dim=-1)
    base_log_probs = torch.log_softmax(base_logits[supervised], dim=-1)
    negative = torch.finfo(style_log_probs.dtype).min
    style_mass = torch.logsumexp(
        style_log_probs.masked_fill(~selected_preferences, negative), dim=-1
    )
    base_mass = torch.logsumexp(
        base_log_probs.masked_fill(~selected_preferences, negative), dim=-1
    )
    lift = style_mass - base_mass
    loss = torch.relu(torch.as_tensor(margin, device=lift.device) - lift).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Preferred action margin loss is not finite")
    return loss, float(lift.mean().detach()), int(supervised.sum())


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
    def __init__(
        self,
        *args,
        beta_kl: float,
        beta_kl_inside: float | None = None,
        beta_kl_outside: float | None = None,
        eta_aux: float = 0.0,
        eta_preference: float = 0.0,
        preference_margin: float = 0.0,
        rho_residual: float = 0.0,
        **kwargs,
    ) -> None:
        beta_inside = beta_kl if beta_kl_inside is None else beta_kl_inside
        beta_outside = beta_kl if beta_kl_outside is None else beta_kl_outside
        if (
            beta_kl < 0
            or beta_inside < 0
            or beta_outside < 0
            or eta_aux < 0
            or eta_preference < 0
            or preference_margin < 0
            or rho_residual < 0
        ):
            raise ValueError("Style loss coefficients must be nonnegative")
        super().__init__(*args, **kwargs)
        self.beta_kl = beta_kl
        self.beta_kl_inside = beta_inside
        self.beta_kl_outside = beta_outside
        self.eta_aux = eta_aux
        self.eta_preference = eta_preference
        self.preference_margin = preference_margin
        self.rho_residual = rho_residual
        self.last_style_kl = 0.0
        self.last_style_kl_inside = 0.0
        self.last_style_kl_outside = 0.0
        self.last_residual_magnitude = 0.0
        self.last_auxiliary_loss = 0.0
        self.last_teacher_agreement = 0.0
        self.last_supervised_tokens = 0
        self.last_preference_loss = 0.0
        self.last_preferred_probability_lift = 0.0
        self.last_opportunity_tokens = 0

    @classmethod
    def create(
        cls,
        model: torch.nn.Module,
        *,
        beta_kl: float,
        beta_kl_inside: float | None = None,
        beta_kl_outside: float | None = None,
        eta_aux: float = 0.0,
        eta_preference: float = 0.0,
        preference_margin: float = 0.0,
        rho_residual: float = 0.0,
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
            beta_kl_inside=beta_kl_inside,
            beta_kl_outside=beta_kl_outside,
            eta_aux=eta_aux,
            eta_preference=eta_preference,
            preference_margin=preference_margin,
            rho_residual=rho_residual,
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
        style_kl_inside = output.logits.sum() * 0.0
        style_kl_outside = output.logits.sum() * 0.0
        inside_count = 0
        outside_count = int(batch.loss_mask.sum())
        if batch.opportunity_mask is None:
            style_kl_outside = categorical_style_kl(
                output.logits, output.base_logits, batch.loss_mask
            )
        else:
            style_kl_inside, style_kl_outside, inside_count, outside_count = (
                partitioned_style_kl(
                    output.logits,
                    output.base_logits,
                    batch.loss_mask,
                    batch.opportunity_mask,
                )
            )
        total_count = inside_count + outside_count
        style_kl = (
            style_kl_inside * inside_count + style_kl_outside * outside_count
        ) / total_count
        self.last_style_kl = float(style_kl.detach())
        self.last_style_kl_inside = float(style_kl_inside.detach())
        self.last_style_kl_outside = float(style_kl_outside.detach())
        residual_magnitude = (
            (output.logits - output.base_logits).square().sum(dim=-1)[
                batch.loss_mask
            ].mean()
        )
        if not bool(torch.isfinite(residual_magnitude)):
            raise FloatingPointError("Style residual magnitude is not finite")
        self.last_residual_magnitude = float(residual_magnitude.detach())
        auxiliary = output.logits.sum() * 0.0
        self.last_auxiliary_loss = 0.0
        self.last_teacher_agreement = 0.0
        self.last_supervised_tokens = 0
        preference = output.logits.sum() * 0.0
        self.last_preference_loss = 0.0
        self.last_preferred_probability_lift = 0.0
        self.last_opportunity_tokens = inside_count
        if self.eta_preference > 0:
            if batch.opportunity_mask is None or batch.preferred_action_mask is None:
                raise ValueError("Style PPO batch is missing opportunity supervision")
            supervised_preference = batch.opportunity_mask & batch.loss_mask
            if bool(supervised_preference.any()):
                preference, lift, count = preferred_action_set_margin_loss(
                    output.logits,
                    output.base_logits,
                    batch.preferred_action_mask,
                    supervised_preference,
                    margin=self.preference_margin,
                )
                self.last_preference_loss = float(preference.detach())
                self.last_preferred_probability_lift = lift
                self.last_opportunity_tokens = count
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
                + self.beta_kl_inside * style_kl_inside
                + self.beta_kl_outside * style_kl_outside
                + self.rho_residual * residual_magnitude
                + self.eta_aux * auxiliary
                + self.eta_preference * preference
            )
        )
