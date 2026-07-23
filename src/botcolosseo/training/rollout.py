from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, fields

import torch

from botcolosseo.training.gae import generalized_advantage_estimate
from botcolosseo.training.ppo import PPOBatch


@dataclass
class RolloutStep:
    frames: torch.Tensor
    scalars: torch.Tensor
    previous_actions: torch.Tensor
    masks: torch.Tensor
    privileged: torch.Tensor
    hidden: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    teacher_actions: torch.Tensor | None = None
    teacher_mask: torch.Tensor | None = None
    route_modes: torch.Tensor | None = None


@dataclass(frozen=True)
class RecurrentRollout:
    frames: torch.Tensor
    scalars: torch.Tensor
    previous_actions: torch.Tensor
    masks: torch.Tensor
    privileged: torch.Tensor
    hidden: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    valid: torch.Tensor
    teacher_actions: torch.Tensor | None = None
    teacher_mask: torch.Tensor | None = None
    route_modes: torch.Tensor | None = None

    def sequence_minibatches(
        self,
        *,
        sequence_length: int,
        burn_in: int,
        minibatch_sequences: int,
        seed: int,
        epoch: int,
        shuffle: bool = True,
    ) -> Iterator[PPOBatch]:
        if sequence_length <= 0 or burn_in < 0 or minibatch_sequences <= 0:
            raise ValueError("Invalid recurrent minibatch settings")
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        environments, time = self.actions.shape
        chunks = [
            (environment, loss_start)
            for environment in range(environments)
            for loss_start in range(0, time, sequence_length)
        ]
        if shuffle:
            generator = torch.Generator().manual_seed(seed + epoch)
            order = torch.randperm(len(chunks), generator=generator).tolist()
            chunks = [chunks[index] for index in order]
        for start in range(0, len(chunks), minibatch_sequences):
            yield self._collate_chunks(
                chunks[start : start + minibatch_sequences],
                sequence_length=sequence_length,
                burn_in=burn_in,
            )

    def _collate_chunks(
        self,
        chunks: list[tuple[int, int]],
        *,
        sequence_length: int,
        burn_in: int,
    ) -> PPOBatch:
        width = sequence_length + burn_in
        time = self.actions.shape[1]

        def padded(
            source: torch.Tensor,
            environment: int,
            begin: int,
            end: int,
            *,
            fill: int = 0,
        ):
            shape = (width, *source.shape[2:])
            target = torch.full(
                shape, fill, dtype=source.dtype, device=source.device
            )
            target[: end - begin] = source[environment, begin:end]
            return target

        collected: dict[str, list[torch.Tensor]] = {
            name: []
            for name in (
                "frames",
                "scalars",
                "previous_actions",
                "masks",
                "privileged",
                "actions",
                "log_probs",
                "values",
                "advantages",
                "returns",
            )
        }
        optional = {
            name: getattr(self, name)
            for name in ("teacher_actions", "teacher_mask", "route_modes")
            if getattr(self, name) is not None
        }
        collected.update({name: [] for name in optional})
        initial_hidden: list[torch.Tensor] = []
        loss_masks: list[torch.Tensor] = []
        for environment, loss_start in chunks:
            begin = max(0, loss_start - burn_in)
            end = min(time, loss_start + sequence_length)
            for name in collected:
                source = getattr(self, name)
                collected[name].append(
                    padded(
                        source,
                        environment,
                        begin,
                        end,
                        fill=-1 if name == "route_modes" else 0,
                    )
                )
            initial_hidden.append(self.hidden[environment, begin])
            loss_mask = torch.zeros(width, dtype=torch.bool, device=self.valid.device)
            offset = loss_start - begin
            count = min(sequence_length, time - loss_start)
            loss_mask[offset : offset + count] = self.valid[
                environment, loss_start : loss_start + count
            ]
            loss_masks.append(loss_mask)
        stacked = {name: torch.stack(parts) for name, parts in collected.items()}
        batch = PPOBatch(
            frames=stacked["frames"],
            scalars=stacked["scalars"],
            previous_actions=stacked["previous_actions"],
            masks=stacked["masks"],
            privileged=stacked["privileged"],
            initial_hidden=torch.stack(initial_hidden).unsqueeze(0),
            actions=stacked["actions"],
            old_log_probs=stacked["log_probs"],
            old_values=stacked["values"],
            advantages=stacked["advantages"],
            returns=stacked["returns"],
            loss_mask=torch.stack(loss_masks),
        )
        values = dict(batch.__dict__)
        values.update({name: stacked[name] for name in optional})
        return PPOBatch(**values)


class RolloutBuffer:
    def __init__(self, *, capacity: int, environments: int) -> None:
        if capacity <= 0 or environments <= 0:
            raise ValueError("Rollout capacity and environment count must be positive")
        self.capacity = capacity
        self.environments = environments
        self._steps: list[RolloutStep] = []

    def __len__(self) -> int:
        return len(self._steps)

    def append(self, step: RolloutStep) -> None:
        if len(self) >= self.capacity:
            raise OverflowError("Rollout buffer is full")
        self._validate_step(step)
        self._steps.append(
            RolloutStep(
                **{
                    item.name: (
                        None
                        if getattr(step, item.name) is None
                        else getattr(step, item.name).detach().clone()
                    )
                    for item in fields(step)
                }
            )
        )

    def _validate_step(self, step: RolloutStep) -> None:
        expected = self.environments
        shapes = {
            "frames": (expected, 1, 84, 84),
            "scalars": (expected, 6),
            "previous_actions": (expected,),
            "masks": (expected,),
            "privileged": (expected, 12),
            "hidden": (1, expected, 256),
            "actions": (expected,),
            "rewards": (expected,),
            "terminated": (expected,),
            "truncated": (expected,),
            "log_probs": (expected,),
            "values": (expected,),
            "next_values": (expected,),
            "teacher_actions": (expected,),
            "teacher_mask": (expected,),
            "route_modes": (expected,),
        }
        for item in fields(step):
            tensor = getattr(step, item.name)
            if tensor is None:
                continue
            if tensor.shape != shapes[item.name]:
                raise ValueError(f"Rollout {item.name} has the wrong shape")
        supervision = (
            step.teacher_actions,
            step.teacher_mask,
            step.route_modes,
        )
        if any(value is None for value in supervision) and not all(
            value is None for value in supervision
        ):
            raise ValueError("Rollout style supervision must be complete or absent")
        if step.terminated.dtype != torch.bool or step.truncated.dtype != torch.bool:
            raise ValueError("Rollout boundaries must be boolean")
        if bool((step.terminated & step.truncated).any()):
            raise ValueError("A rollout step cannot terminate and truncate together")
        if not bool(torch.all((step.masks == 0) | (step.masks == 1))):
            raise ValueError("Rollout masks must contain only zero or one")
        if int(step.actions.min()) < 0 or int(step.actions.max()) >= 13:
            raise ValueError("Rollout action is outside the action space")
        if step.teacher_actions is not None:
            if (
                step.teacher_mask is None
                or step.route_modes is None
                or step.teacher_mask.dtype != torch.bool
            ):
                raise ValueError("Invalid rollout style supervision")
            if int(step.teacher_actions.min()) < 0 or int(step.teacher_actions.max()) >= 13:
                raise ValueError("Teacher action is outside the action space")
            if int(step.route_modes.min()) < -1 or int(step.route_modes.max()) > 2:
                raise ValueError("Route mode is outside the supported range")
        devices = {
            tensor.device
            for item in fields(step)
            if (tensor := getattr(step, item.name)) is not None
        }
        if len(devices) != 1:
            raise ValueError("Rollout tensors must share a device")
        floating = (
            step.scalars,
            step.masks,
            step.privileged,
            step.hidden,
            step.rewards,
            step.log_probs,
            step.values,
            step.next_values,
        )
        if any(not bool(torch.isfinite(tensor).all()) for tensor in floating):
            raise FloatingPointError("Rollout values must be finite")

    def finalize(self, *, gamma: float, gae_lambda: float) -> RecurrentRollout:
        if not self._steps:
            raise ValueError("Cannot finalize an empty rollout")

        def stack(name: str) -> torch.Tensor:
            return torch.stack([getattr(step, name) for step in self._steps], dim=1)

        values = stack("values")
        gae = generalized_advantage_estimate(
            stack("rewards"),
            values,
            stack("next_values"),
            stack("terminated"),
            stack("truncated"),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        hidden = torch.stack([step.hidden[0] for step in self._steps], dim=1)
        return RecurrentRollout(
            frames=stack("frames"),
            scalars=stack("scalars"),
            previous_actions=stack("previous_actions"),
            masks=stack("masks"),
            privileged=stack("privileged"),
            hidden=hidden,
            actions=stack("actions"),
            log_probs=stack("log_probs"),
            values=values,
            advantages=gae.advantages,
            returns=gae.returns,
            valid=torch.ones_like(values, dtype=torch.bool),
            teacher_actions=(
                None if self._steps[0].teacher_actions is None else stack("teacher_actions")
            ),
            teacher_mask=(
                None if self._steps[0].teacher_mask is None else stack("teacher_mask")
            ),
            route_modes=(
                None if self._steps[0].route_modes is None else stack("route_modes")
            ),
        )
