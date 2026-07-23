from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch

from botcolosseo.agents.checkpoint import (
    CheckpointMetadata,
    load_training_checkpoint,
    save_training_checkpoint,
)
from botcolosseo.agents.model import AsymmetricActorCritic
from botcolosseo.training.gae import normalize_advantages


@dataclass(frozen=True)
class PPOBatch:
    frames: torch.Tensor
    scalars: torch.Tensor
    previous_actions: torch.Tensor
    masks: torch.Tensor
    privileged: torch.Tensor
    initial_hidden: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    loss_mask: torch.Tensor
    teacher_actions: torch.Tensor | None = None
    teacher_mask: torch.Tensor | None = None
    route_modes: torch.Tensor | None = None


class PPOLoss(NamedTuple):
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approximate_kl: float
    clip_fraction: float
    valid_count: int


class ExcessiveKLError(RuntimeError):
    def __init__(self, approximate_kl: float, max_kl: float) -> None:
        self.approximate_kl = approximate_kl
        self.max_kl = max_kl
        super().__init__(
            f"PPO approximate KL {approximate_kl:.6f} exceeded {max_kl:.6f}"
        )


@dataclass(frozen=True)
class PPOUpdateMetrics:
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    valid_count: int
    pre_clip_grad_norm: float
    post_clip_grad_norm: float
    learning_rate: float
    update: int


def _require_finite(*tensors: torch.Tensor) -> None:
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise FloatingPointError("PPO inputs and outputs must be finite")


def clipped_ppo_loss(
    *,
    new_log_probs: torch.Tensor,
    entropy: torch.Tensor,
    values: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    valid: torch.Tensor,
    policy_clip: float,
    value_clip: float,
    value_coefficient: float,
    entropy_coefficient: float,
    max_kl: float,
) -> PPOLoss:
    expected = new_log_probs.shape
    tensors = (entropy, values, old_log_probs, old_values, advantages, returns)
    if any(tensor.shape != expected for tensor in tensors) or valid.shape != expected:
        raise ValueError("PPO statistic tensors must share a shape")
    if valid.dtype != torch.bool or not bool(valid.any()):
        raise ValueError("PPO valid mask must select at least one item")
    if policy_clip <= 0 or value_clip <= 0 or max_kl <= 0:
        raise ValueError("PPO clipping and KL limits must be positive")
    if value_coefficient < 0 or entropy_coefficient < 0:
        raise ValueError("PPO loss coefficients must be nonnegative")
    _require_finite(new_log_probs, *tensors)

    selected_new = new_log_probs[valid]
    selected_old = old_log_probs[valid]
    selected_advantages = advantages[valid]
    log_ratio = selected_new - selected_old
    ratio = log_ratio.exp()
    unclipped_policy = ratio * selected_advantages
    clipped_policy = ratio.clamp(1.0 - policy_clip, 1.0 + policy_clip)
    clipped_policy = clipped_policy * selected_advantages
    policy_loss = -torch.minimum(unclipped_policy, clipped_policy).mean()

    selected_values = values[valid]
    selected_old_values = old_values[valid]
    selected_returns = returns[valid]
    clipped_values = selected_old_values + (selected_values - selected_old_values).clamp(
        -value_clip, value_clip
    )
    value_error = (selected_values - selected_returns).square()
    clipped_value_error = (clipped_values - selected_returns).square()
    value_loss = 0.5 * torch.maximum(value_error, clipped_value_error).mean()
    mean_entropy = entropy[valid].mean()
    total_loss = (
        policy_loss
        + value_coefficient * value_loss
        - entropy_coefficient * mean_entropy
    )
    approximate_kl_tensor = ((ratio - 1.0) - log_ratio).mean()
    _require_finite(total_loss, approximate_kl_tensor)
    approximate_kl = float(approximate_kl_tensor.detach())
    if approximate_kl > max_kl:
        raise ExcessiveKLError(approximate_kl, max_kl)
    clip_fraction = float(((ratio - 1.0).abs() > policy_clip).float().mean())
    return PPOLoss(
        total_loss,
        policy_loss,
        value_loss,
        mean_entropy,
        approximate_kl,
        clip_fraction,
        int(valid.sum()),
    )


def ppo_loss(
    *,
    logits: torch.Tensor,
    values: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    valid: torch.Tensor,
    policy_clip: float,
    value_clip: float,
    value_coefficient: float,
    entropy_coefficient: float,
    max_kl: float,
) -> PPOLoss:
    if logits.shape[:-1] != actions.shape or values.shape != actions.shape:
        raise ValueError("PPO model outputs and actions are incompatible")
    _require_finite(logits)
    distribution = torch.distributions.Categorical(logits=logits)
    return clipped_ppo_loss(
        new_log_probs=distribution.log_prob(actions),
        entropy=distribution.entropy(),
        values=values,
        old_log_probs=old_log_probs,
        old_values=old_values,
        advantages=advantages,
        returns=returns,
        valid=valid,
        policy_clip=policy_clip,
        value_clip=value_clip,
        value_coefficient=value_coefficient,
        entropy_coefficient=entropy_coefficient,
        max_kl=max_kl,
    )


class PPOTrainer:
    def __init__(
        self,
        model: AsymmetricActorCritic,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        *,
        gradient_clip: float,
        policy_clip: float,
        value_clip: float,
        value_coefficient: float,
        entropy_coefficient: float,
        max_kl: float,
    ) -> None:
        if gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.policy_clip = policy_clip
        self.value_clip = value_clip
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.max_kl = max_kl
        self.updates = 0

    @classmethod
    def create(
        cls,
        model: AsymmetricActorCritic,
        *,
        learning_rate: float,
        total_updates: int,
        gradient_clip: float,
        policy_clip: float,
        value_clip: float,
        value_coefficient: float,
        entropy_coefficient: float,
        max_kl: float,
        weight_decay: float = 0.0,
    ) -> PPOTrainer:
        if learning_rate <= 0 or total_updates <= 0 or weight_decay < 0:
            raise ValueError("Invalid PPO optimizer settings")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_updates
        )
        return cls(
            model,
            optimizer,
            scheduler,
            gradient_clip=gradient_clip,
            policy_clip=policy_clip,
            value_clip=value_clip,
            value_coefficient=value_coefficient,
            entropy_coefficient=entropy_coefficient,
            max_kl=max_kl,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _move(self, batch: PPOBatch) -> PPOBatch:
        return PPOBatch(
            **{
                name: None if value is None else value.to(self.device)
                for name, value in batch.__dict__.items()
            }
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
        return ppo_loss(
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

    def _metrics(
        self, loss: PPOLoss, *, pre_clip: float = 0.0, post_clip: float = 0.0
    ) -> PPOUpdateMetrics:
        return PPOUpdateMetrics(
            total_loss=float(loss.total_loss.detach()),
            policy_loss=float(loss.policy_loss.detach()),
            value_loss=float(loss.value_loss.detach()),
            entropy=float(loss.entropy.detach()),
            approximate_kl=loss.approximate_kl,
            clip_fraction=loss.clip_fraction,
            valid_count=loss.valid_count,
            pre_clip_grad_norm=pre_clip,
            post_clip_grad_norm=post_clip,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            update=self.updates,
        )

    @torch.no_grad()
    def evaluate(self, batch: PPOBatch) -> PPOUpdateMetrics:
        self.model.eval()
        return self._metrics(self._loss(self._move(batch)))

    def train_step(self, batch: PPOBatch) -> PPOUpdateMetrics:
        self.model.train()
        moved = self._move(batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._loss(moved)
        loss.total_loss.backward()
        pre_clip = float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        )
        squared = sum(
            float(parameter.grad.detach().norm()) ** 2
            for parameter in self.model.parameters()
            if parameter.grad is not None
        )
        post_clip = math.sqrt(squared)
        if not math.isfinite(pre_clip) or not math.isfinite(post_clip):
            raise FloatingPointError("PPO gradient norm is not finite")
        self.optimizer.step()
        self.scheduler.step()
        self.updates += 1
        return self._metrics(loss, pre_clip=pre_clip, post_clip=post_clip)

    def save(
        self,
        path: Path,
        *,
        config_hash: str,
        scenario_hash: str,
        counters: dict[str, int] | None = None,
    ) -> Path:
        metadata_counters = dict(counters or {})
        if "updates" in metadata_counters:
            raise ValueError("PPO checkpoint counters reserve the updates key")
        metadata_counters["updates"] = self.updates
        return save_training_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metadata=CheckpointMetadata(
                config_hash=config_hash,
                scenario_hash=scenario_hash,
                counters=metadata_counters,
            ),
        )

    def load(
        self,
        path: Path,
        *,
        config_hash: str,
        scenario_hash: str,
        restore_rng: bool,
    ) -> CheckpointMetadata:
        metadata = load_training_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            expected_config_hash=config_hash,
            expected_scenario_hash=scenario_hash,
            restore_rng=restore_rng,
        )
        self.updates = metadata.counters["updates"]
        return metadata
