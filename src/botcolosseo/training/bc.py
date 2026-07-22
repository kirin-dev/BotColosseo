from __future__ import annotations

import json
import math
import random
from collections import Counter
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
from botcolosseo.agents.duel_teachers import create_duel_teacher
from botcolosseo.agents.model import RecurrentActor
from botcolosseo.data.demonstrations import load_demonstration_shard
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_types import DuelActorObservation
from botcolosseo.envs.synchronous_duel import SynchronousDuelEnv
from botcolosseo.scenarios.duel_splits import DuelCase
from botcolosseo.scenarios.regions import RegionGraph


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


class DemonstrationChunkDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        shard_paths: tuple[Path, ...],
        *,
        chunk_length: int,
        max_transitions: int | None = None,
        allow_masked: bool = False,
    ) -> None:
        if not shard_paths or chunk_length <= 0:
            raise ValueError("shard_paths and chunk_length must be nonempty")
        if max_transitions is not None and max_transitions <= 0:
            raise ValueError("max_transitions must be positive")
        loaded: dict[str, list[np.ndarray]] = {}
        remaining = max_transitions
        for path in shard_paths:
            arrays = load_demonstration_shard(path, require_all_valid=not allow_masked)
            take = (
                len(arrays["frame"])
                if remaining is None
                else min(remaining, len(arrays["frame"]))
            )
            for name, array in arrays.items():
                loaded.setdefault(name, []).append(array[:take])
            if remaining is not None:
                remaining -= take
                if remaining == 0:
                    break
        self._arrays = {
            name: np.concatenate(parts, axis=0) for name, parts in loaded.items()
        }
        self.chunk_length = chunk_length
        self.transition_count = len(self._arrays["frame"])
        starts = tuple(range(0, self.transition_count, chunk_length))
        if allow_masked:
            starts = tuple(
                start
                for start in starts
                if bool(
                    np.any(self._arrays["valid_mask"][start : start + chunk_length])
                )
            )
        self._starts = starts
        if not self._starts:
            raise ValueError("Demonstration dataset contains no supervised chunks")

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = self._starts[index]
        stop = min(start + self.chunk_length, self.transition_count)
        size = stop - start
        frames = np.zeros((self.chunk_length, 1, 84, 84), dtype=np.uint8)
        scalars = np.zeros((self.chunk_length, 6), dtype=np.float32)
        previous_actions = np.zeros(self.chunk_length, dtype=np.int64)
        actions = np.zeros(self.chunk_length, dtype=np.int64)
        valid = np.zeros(self.chunk_length, dtype=np.bool_)
        present = np.zeros(self.chunk_length, dtype=np.bool_)
        masks = np.zeros(self.chunk_length, dtype=np.float32)
        frames[:size, 0] = self._arrays["frame"][start:stop]
        scalars[:size] = self._arrays["scalars"][start:stop]
        previous_actions[:size] = self._arrays["previous_action"][start:stop]
        actions[:size] = self._arrays["teacher_action"][start:stop]
        valid[:size] = self._arrays["valid_mask"][start:stop]
        present[:size] = True
        episode_start = self._arrays["episode_start"][start:stop]
        masks[:size] = (~episode_start).astype(np.float32)
        masks[0] = 0.0
        return {
            "frames": torch.from_numpy(frames),
            "scalars": torch.from_numpy(scalars),
            "previous_actions": torch.from_numpy(previous_actions),
            "actions": torch.from_numpy(actions),
            "masks": torch.from_numpy(masks),
            "valid": torch.from_numpy(valid),
            "present": torch.from_numpy(present),
        }


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


class BestCheckpointTracker:
    def __init__(self) -> None:
        self.best_objective_rate = float("-inf")
        self.best_validation_loss = float("inf")
        self.best_update: int | None = None

    def update(
        self, *, validation_loss: float, objective_rate: float, update: int
    ) -> bool:
        if not math.isfinite(validation_loss) or not 0.0 <= objective_rate <= 1.0:
            raise ValueError("Invalid validation selection metric")
        better = objective_rate > self.best_objective_rate or (
            objective_rate == self.best_objective_rate
            and validation_loss < self.best_validation_loss
        )
        if better:
            self.best_objective_rate = objective_rate
            self.best_validation_loss = validation_loss
            self.best_update = update
        return better


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

    def save(self, path: Path, *, config_hash: str, scenario_hash: str) -> Path:
        return save_training_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metadata=CheckpointMetadata(
                config_hash=config_hash,
                scenario_hash=scenario_hash,
                counters={"updates": self.updates},
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


def load_shard_paths(manifest_path: Path) -> tuple[Path, ...]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("split") not in {"train", "validation"}:
        raise ValueError("BC may load train or validation manifests only")
    if payload.get("test_cases_accessed") is not False:
        raise ValueError("BC manifest must certify that test cases were not accessed")
    paths = tuple(manifest_path.parent / item["file"] for item in payload["shards"])
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("BC manifest references a missing shard")
    return paths


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def observation_tensors(
    observation: DuelActorObservation,
    *,
    episode_start: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scalars = torch.tensor(
        [
            observation.health / 200.0,
            observation.armor / 200.0,
            min(observation.ammo, 100.0) / 100.0,
            min(observation.own_score, 3) / 3.0,
            min(observation.opponent_score, 3) / 3.0,
            float(observation.has_core),
        ],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 1, 6)
    frame = torch.from_numpy(np.array(observation.frame, copy=True)).to(device)
    frame = frame.reshape(1, 1, 1, 84, 84)
    previous_action = torch.tensor(
        [[observation.previous_action]], dtype=torch.long, device=device
    )
    mask = torch.tensor(
        [[0.0 if episode_start else 1.0]], dtype=torch.float32, device=device
    )
    return frame, scalars, previous_action, mask


@torch.no_grad()
def evaluate_closed_loop_episode(
    model: RecurrentActor,
    *,
    root: Path,
    case: DuelCase,
    max_decisions: int = 525,
) -> dict[str, object]:
    if case.split != "validation":
        raise ValueError("Closed-loop BC selection may use validation cases only")
    graph = RegionGraph.from_yaml(
        root / "assets/scenarios/crystal_run/src/regions.yaml"
    )
    env = SynchronousDuelEnv(
        config_path=root / "assets/scenarios/crystal_run/crystal_run.cfg",
        region_graph=graph,
        seed=case.seed,
        max_decisions=max_decisions,
    )
    opponent_side = "opponent" if case.learner_side == "host" else "host"
    opponent = create_duel_teacher(case.opponent, graph, side=opponent_side)
    device = next(model.parameters()).device
    previous_mode = model.training
    model.eval()
    events: Counter[str] = Counter()
    try:
        observations, _ = env.reset()
        opponent.reset(seed=case.seed)
        hidden = model.initial_state(1, device=device)
        episode_start = True
        decisions = 0
        terminated = False
        truncated = False
        initial_score = (
            observations.host.own_score
            if case.learner_side == "host"
            else observations.opponent.own_score
        )
        while decisions < max_decisions and not (terminated or truncated):
            observation = (
                observations.host
                if case.learner_side == "host"
                else observations.opponent
            )
            tensors = observation_tensors(
                observation, episode_start=episode_start, device=device
            )
            output = model(*tensors, hidden)
            learner_action = MacroAction(int(output.logits[0, 0].argmax()))
            hidden = output.hidden
            opponent_action = opponent.act(env.teacher_state())
            host_action, away_action = (
                (learner_action, opponent_action)
                if case.learner_side == "host"
                else (opponent_action, learner_action)
            )
            step = env.step(host_action, away_action)
            observations = type(observations)(step.host, step.opponent)
            events.update(f"{event.side}:{event.type.value}" for event in step.events)
            decisions += 1
            episode_start = False
            terminated, truncated = step.terminated, step.truncated
        final_observation = (
            observations.host if case.learner_side == "host" else observations.opponent
        )
        objective_completed = final_observation.own_score > initial_score
        return {
            "case_seed": case.seed,
            "decisions": decisions,
            "event_counts": dict(sorted(events.items())),
            "learner_side": case.learner_side,
            "objective_completed": objective_completed,
            "opponent": case.opponent,
            "terminated": terminated,
            "truncated": truncated,
        }
    finally:
        env.close()
        model.train(previous_mode)
