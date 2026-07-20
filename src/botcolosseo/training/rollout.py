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

        def padded(source: torch.Tensor, environment: int, begin: int, end: int):
            shape = (width, *source.shape[2:])
            target = torch.zeros(shape, dtype=source.dtype, device=source.device)
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
        initial_hidden: list[torch.Tensor] = []
        loss_masks: list[torch.Tensor] = []
        for environment, loss_start in chunks:
            begin = max(0, loss_start - burn_in)
            end = min(time, loss_start + sequence_length)
            for name in collected:
                collected[name].append(
                    padded(getattr(self, name), environment, begin, end)
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
        return PPOBatch(
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
                    item.name: getattr(step, item.name).detach().clone()
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
        }
        for item in fields(step):
            tensor = getattr(step, item.name)
            if tensor.shape != shapes[item.name]:
                raise ValueError(f"Rollout {item.name} has the wrong shape")
        if step.terminated.dtype != torch.bool or step.truncated.dtype != torch.bool:
            raise ValueError("Rollout boundaries must be boolean")
        if bool((step.terminated & step.truncated).any()):
            raise ValueError("A rollout step cannot terminate and truncate together")
        if not bool(torch.all((step.masks == 0) | (step.masks == 1))):
            raise ValueError("Rollout masks must contain only zero or one")
        if int(step.actions.min()) < 0 or int(step.actions.max()) >= 13:
            raise ValueError("Rollout action is outside the action space")
        devices = {getattr(step, item.name).device for item in fields(step)}
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
        )
