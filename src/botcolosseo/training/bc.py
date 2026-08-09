from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from botcolosseo.agents.checkpoint import (
    CheckpointMetadata,
    load_training_checkpoint,
    save_training_checkpoint,
)
from botcolosseo.agents.model import RecurrentActor


class BCLossMetrics(NamedTuple):
    loss: torch.Tensor
    accuracy: float
    valid_count: int


@dataclass(frozen=True)
class BCStepMetrics:
    loss: float
    accuracy: float
    valid_count: int
    pre_clip_grad_norm: float
    post_clip_grad_norm: float
    learning_rate: float
    update: int


@dataclass(frozen=True)
class BCEvaluationMetrics:
    loss: float
    accuracy: float
    valid_count: int


class DeterministicBatchStream:
    def __init__(
        self,
        dataset: Dataset[dict[str, torch.Tensor]],
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        if len(dataset) <= 0 or batch_size <= 0:
            raise ValueError("dataset and batch_size must be nonempty")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.batches_per_epoch = math.ceil(len(dataset) / batch_size)

    def batch(self, update: int) -> dict[str, torch.Tensor]:
        if update < 0:
            raise ValueError("update must be nonnegative")
        epoch, offset = divmod(update, self.batches_per_epoch)
        generator = torch.Generator().manual_seed(self.seed + epoch)
        order = torch.randperm(len(self.dataset), generator=generator).tolist()
        start = offset * self.batch_size
        indices = order[start : start + self.batch_size]
        return default_collate([self.dataset[index] for index in indices])


def make_validation_loader(
    dataset: Dataset[dict[str, torch.Tensor]], *, batch_size: int
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def behavior_cloning_metrics(
    logits: torch.Tensor, actions: torch.Tensor, valid: torch.Tensor
) -> BCLossMetrics:
    if logits.shape[:-1] != actions.shape or actions.shape != valid.shape:
        raise ValueError("BC logits, actions, and valid mask have incompatible shapes")
    selected_logits = logits[valid]
    selected_actions = actions[valid]
    if selected_actions.numel() == 0:
        raise ValueError("BC batch contains no valid transitions")
    loss = torch.nn.functional.cross_entropy(selected_logits, selected_actions)
    accuracy = float(
        (selected_logits.argmax(dim=-1) == selected_actions).float().mean().detach()
    )
    return BCLossMetrics(loss, accuracy, int(selected_actions.numel()))


class BCTrainer:
    def __init__(
        self,
        model: RecurrentActor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        *,
        gradient_clip: float,
    ) -> None:
        if gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.updates = 0

    @classmethod
    def create(
        cls,
        model: RecurrentActor,
        *,
        learning_rate: float,
        gradient_clip: float,
        total_updates: int,
        weight_decay: float = 0.0,
    ) -> BCTrainer:
        if learning_rate <= 0 or total_updates <= 0 or weight_decay < 0:
            raise ValueError("Invalid BC optimizer settings")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_updates
        )
        return cls(model, optimizer, scheduler, gradient_clip=gradient_clip)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _move(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: tensor.to(self.device) for name, tensor in batch.items()}

    def train_step(self, batch: dict[str, torch.Tensor]) -> BCStepMetrics:
        self.model.train()
        moved = self._move(batch)
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(
            moved["frames"],
            moved["scalars"],
            moved["previous_actions"],
            moved["masks"],
        )
        metrics = behavior_cloning_metrics(
            output.logits, moved["actions"], moved["valid"]
        )
        metrics.loss.backward()
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
            raise FloatingPointError("BC gradient norm is not finite")
        self.optimizer.step()
        self.scheduler.step()
        self.updates += 1
        return BCStepMetrics(
            loss=float(metrics.loss.detach()),
            accuracy=metrics.accuracy,
            valid_count=metrics.valid_count,
            pre_clip_grad_norm=pre_clip,
            post_clip_grad_norm=post_clip,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            update=self.updates,
        )

    @torch.no_grad()
    def evaluate_batch(self, batch: dict[str, torch.Tensor]) -> BCEvaluationMetrics:
        self.model.eval()
        moved = self._move(batch)
        output = self.model(
            moved["frames"],
            moved["scalars"],
            moved["previous_actions"],
            moved["masks"],
        )
        metrics = behavior_cloning_metrics(
            output.logits, moved["actions"], moved["valid"]
        )
        return BCEvaluationMetrics(
            float(metrics.loss), metrics.accuracy, metrics.valid_count
        )

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> BCEvaluationMetrics:
        weighted_loss = 0.0
        weighted_accuracy = 0.0
        total = 0
        for batch in loader:
            metrics = self.evaluate_batch(batch)
            weighted_loss += metrics.loss * metrics.valid_count
            weighted_accuracy += metrics.accuracy * metrics.valid_count
            total += metrics.valid_count
        if total == 0:
            raise ValueError("Validation loader contains no valid transitions")
        return BCEvaluationMetrics(
            weighted_loss / total, weighted_accuracy / total, total
        )

    def save(
        self,
        path: Path,
        *,
        config_hash: str,
        scenario_hash: str,
        lineage: dict[str, str | int] | None = None,
    ) -> Path:
        return save_training_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metadata=CheckpointMetadata(
                config_hash=config_hash,
                scenario_hash=scenario_hash,
                counters={"updates": self.updates},
                lineage=dict(lineage or {}),
            ),
        )

    def load(
        self,
        path: Path,
        *,
        config_hash: str,
        scenario_hash: str,
        restore_rng: bool,
    ) -> None:
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


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(payload, sort_keys=True) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
