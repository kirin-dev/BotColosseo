from __future__ import annotations

import math

import torch

from botcolosseo.training.gae import normalize_advantages
from botcolosseo.training.ppo import (
    PPOBatch,
    PPOLoss,
    PPOTrainer,
    ppo_loss,
)


def teacher_coefficient_at_step(environment_steps: int) -> float:
    if environment_steps < 0:
        raise ValueError("Environment steps must be nonnegative")
    if environment_steps <= 100_000:
        return 1.0
    if environment_steps < 600_000:
        progress = (environment_steps - 100_000) / 500_000
        return 1.0 - 0.8 * progress
    return 0.2


def main_learning_rate_at_step(
    environment_steps: int,
    *,
    total_steps: int,
    initial_rate: float,
    final_rate: float,
) -> float:
    if not 0 <= environment_steps <= total_steps:
        raise ValueError("Environment step is outside the learning-rate schedule")
    progress = environment_steps / total_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_rate + (initial_rate - final_rate) * cosine


def visual_learning_rate_at_step(environment_steps: int) -> float:
    if environment_steps < 0:
        raise ValueError("Environment steps must be nonnegative")
    if environment_steps <= 600_000:
        return 0.0
    if environment_steps < 620_000:
        return 5e-7 * (environment_steps - 600_000) / 20_000
    progress = min((environment_steps - 620_000) / 380_000, 1.0)
    return 1e-7 + (5e-7 - 1e-7) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


class EnvironmentStepLRScheduler:
    """Checkpointable LR schedule controlled by committed environment steps."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        main_initial_rate: float,
        main_final_rate: float,
    ) -> None:
        if len(optimizer.param_groups) != 2:
            raise ValueError("Visual curriculum optimizer requires two groups")
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.main_initial_rate = main_initial_rate
        self.main_final_rate = main_final_rate
        self.environment_steps = 0
        self.set_environment_steps(0)

    def set_environment_steps(self, environment_steps: int) -> None:
        self.optimizer.param_groups[0]["lr"] = main_learning_rate_at_step(
            environment_steps,
            total_steps=self.total_steps,
            initial_rate=self.main_initial_rate,
            final_rate=self.main_final_rate,
        )
        self.optimizer.param_groups[1]["lr"] = visual_learning_rate_at_step(
            environment_steps
        )
        self.environment_steps = environment_steps

    def step(self) -> None:
        return None

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, float | int]:
        return {
            "environment_steps": self.environment_steps,
            "total_steps": self.total_steps,
            "main_initial_rate": self.main_initial_rate,
            "main_final_rate": self.main_final_rate,
        }

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        expected = {
            "total_steps": self.total_steps,
            "main_initial_rate": self.main_initial_rate,
            "main_final_rate": self.main_final_rate,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("Environment-step LR scheduler identity changed")
        self.set_environment_steps(int(state["environment_steps"]))


class TeacherAnchoredPPOTrainer(PPOTrainer):
    """PPO with an on-policy Strong Teacher anchor."""

    def __init__(
        self,
        *args,
        teacher_coefficient: float,
        reference_actor: torch.nn.Module | None = None,
        reference_kl_coefficient: float = 0.0,
        replay_coefficient: float = 0.0,
        **kwargs,
    ) -> None:
        if min(
            teacher_coefficient,
            reference_kl_coefficient,
            replay_coefficient,
        ) < 0:
            raise ValueError("Conservative PPO coefficients must be nonnegative")
        if reference_kl_coefficient > 0 and reference_actor is None:
            raise ValueError("Reference KL requires a frozen BC Actor")
        super().__init__(*args, **kwargs)
        if reference_actor is not None:
            reference_actor.requires_grad_(False)
            reference_actor.eval()
            optimized = {
                id(parameter)
                for group in self.optimizer.param_groups
                for parameter in group["params"]
            }
            if any(id(parameter) in optimized for parameter in reference_actor.parameters()):
                raise ValueError("Frozen BC reference is present in the optimizer")
        self.reference_actor = reference_actor
        self.reference_kl_coefficient = reference_kl_coefficient
        self.replay_coefficient = replay_coefficient
        self._teacher_coefficient_scale = teacher_coefficient
        self.teacher_coefficient = teacher_coefficient
        self.last_teacher_loss = 0.0
        self.last_teacher_agreement = 0.0
        self.last_supervised_tokens = 0
        self.last_reference_kl = 0.0
        self.last_replay_loss = 0.0
        self.last_replay_agreement = 0.0
        self.last_replay_tokens = 0

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
        visual_parameters: tuple[torch.nn.Parameter, ...] = (),
        total_environment_steps: int | None = None,
        final_learning_rate: float = 0.0,
        reference_actor: torch.nn.Module | None = None,
        reference_kl_coefficient: float = 0.0,
        replay_coefficient: float = 0.0,
    ) -> TeacherAnchoredPPOTrainer:
        if learning_rate <= 0 or total_updates <= 0 or weight_decay < 0:
            raise ValueError("Invalid teacher-anchored PPO optimizer settings")
        if visual_parameters:
            if total_environment_steps is None or final_learning_rate <= 0:
                raise ValueError("Visual curriculum schedule is incomplete")
            visual_ids = {id(parameter) for parameter in visual_parameters}
            main_parameters = tuple(
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad and id(parameter) not in visual_ids
            )
            optimizer = torch.optim.AdamW(
                (
                    {"params": main_parameters, "lr": learning_rate},
                    {"params": visual_parameters, "lr": 0.0},
                ),
                weight_decay=weight_decay,
            )
            scheduler = EnvironmentStepLRScheduler(
                optimizer,
                total_steps=total_environment_steps,
                main_initial_rate=learning_rate,
                main_final_rate=final_learning_rate,
            )
        else:
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
            reference_actor=reference_actor,
            reference_kl_coefficient=reference_kl_coefficient,
            replay_coefficient=replay_coefficient,
            gradient_clip=gradient_clip,
            policy_clip=policy_clip,
            value_clip=value_clip,
            value_coefficient=value_coefficient,
            entropy_coefficient=entropy_coefficient,
            max_kl=max_kl,
        )

    def set_environment_steps(self, environment_steps: int) -> None:
        if isinstance(self.scheduler, EnvironmentStepLRScheduler):
            self.scheduler.set_environment_steps(environment_steps)
            self.teacher_coefficient = (
                self._teacher_coefficient_scale
                * teacher_coefficient_at_step(environment_steps)
            )

    def _loss(
        self,
        batch: PPOBatch,
        replay_batch: dict[str, torch.Tensor] | None = None,
    ) -> PPOLoss:
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
        reference_kl = torch.zeros((), device=output.logits.device)
        if self.reference_kl_coefficient > 0:
            if self.reference_actor is None:
                raise RuntimeError("Conservative PPO reference Actor is missing")
            with torch.no_grad():
                reference = self.reference_actor(
                    batch.frames,
                    batch.scalars,
                    batch.previous_actions,
                    batch.masks,
                    batch.initial_hidden,
                )
            reference_log_probs = torch.nn.functional.log_softmax(
                reference.logits[batch.loss_mask],
                dim=-1,
            )
            current_log_probs = torch.nn.functional.log_softmax(
                output.logits[batch.loss_mask],
                dim=-1,
            )
            reference_probs = reference_log_probs.exp()
            reference_kl = (
                reference_probs * (reference_log_probs - current_log_probs)
            ).sum(dim=-1).mean()
            if not bool(torch.isfinite(reference_kl)):
                raise FloatingPointError("Frozen-reference KL is not finite")
        self.last_reference_kl = float(reference_kl.detach())

        replay_loss = torch.zeros((), device=output.logits.device)
        if self.replay_coefficient > 0:
            if replay_batch is None:
                raise ValueError("Conservative PPO replay batch is missing")
            required = {
                "frames",
                "scalars",
                "previous_actions",
                "masks",
                "actions",
                "valid",
            }
            if not required.issubset(replay_batch):
                raise ValueError("Conservative PPO replay batch is incomplete")
            replay = self.model.actor(
                replay_batch["frames"],
                replay_batch["scalars"],
                replay_batch["previous_actions"],
                replay_batch["masks"],
            )
            replay_valid = replay_batch["valid"]
            if replay_valid.dtype != torch.bool or not bool(replay_valid.any()):
                raise ValueError("Conservative PPO replay selects no valid tokens")
            replay_logits = replay.logits[replay_valid]
            replay_actions = replay_batch["actions"][replay_valid]
            replay_loss = torch.nn.functional.cross_entropy(
                replay_logits,
                replay_actions,
            )
            if not bool(torch.isfinite(replay_loss)):
                raise FloatingPointError("Conservative PPO replay loss is not finite")
            self.last_replay_agreement = float(
                (replay_logits.argmax(dim=-1) == replay_actions)
                .float()
                .mean()
                .detach()
            )
            self.last_replay_tokens = int(replay_actions.numel())
        else:
            self.last_replay_agreement = 0.0
            self.last_replay_tokens = 0
        self.last_replay_loss = float(replay_loss.detach())
        return loss._replace(
            total_loss=(
                loss.total_loss + self.teacher_coefficient * teacher_loss
                + self.reference_kl_coefficient * reference_kl
                + self.replay_coefficient * replay_loss
            )
        )
