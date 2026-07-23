from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch

from botcolosseo.agents.style_model import StyledActorCritic
from botcolosseo.data.demonstrations import TASK_IDS
from botcolosseo.envs.actions import MacroAction
from botcolosseo.training.style_ppo import categorical_style_kl


class StyleDistillationLoss(NamedTuple):
    total: torch.Tensor
    imitation: torch.Tensor
    base_kl: torch.Tensor
    accuracy: float
    supervised_count: int


@dataclass(frozen=True)
class StyleDistillationStep:
    total_loss: float
    imitation_loss: float
    base_kl: float
    accuracy: float
    supervised_count: int
    gradient_norm: float
    update: int


def style_distillation_loss(
    style_logits: torch.Tensor,
    base_logits: torch.Tensor,
    actions: torch.Tensor,
    supervised: torch.Tensor,
    present: torch.Tensor,
    *,
    beta_kl: float,
) -> StyleDistillationLoss:
    if beta_kl < 0:
        raise ValueError("Style distillation KL coefficient must be nonnegative")
    if (
        style_logits.shape != base_logits.shape
        or style_logits.shape[:-1] != actions.shape
        or actions.shape != supervised.shape
        or supervised.shape != present.shape
        or supervised.dtype is not torch.bool
        or present.dtype is not torch.bool
    ):
        raise ValueError("Style distillation tensors have incompatible shapes")
    if not torch.any(supervised) or torch.any(supervised & ~present):
        raise ValueError("Style distillation supervision mask is invalid")
    selected_logits = style_logits[supervised]
    selected_actions = actions[supervised]
    imitation = torch.nn.functional.cross_entropy(selected_logits, selected_actions)
    base_kl = categorical_style_kl(style_logits, base_logits, present)
    accuracy = float((selected_logits.argmax(dim=-1) == selected_actions).float().mean().detach())
    return StyleDistillationLoss(
        imitation + beta_kl * base_kl,
        imitation,
        base_kl,
        accuracy,
        int(selected_actions.numel()),
    )


class StyleDistillationTrainer:
    def __init__(
        self,
        model: StyledActorCritic,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        *,
        gradient_clip: float,
        beta_kl: float,
    ) -> None:
        if gradient_clip <= 0 or beta_kl < 0:
            raise ValueError("Invalid style distillation optimizer settings")
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.beta_kl = beta_kl
        self.updates = 0

    @classmethod
    def create(
        cls,
        model: StyledActorCritic,
        *,
        learning_rate: float,
        weight_decay: float,
        gradient_clip: float,
        beta_kl: float,
        total_updates: int,
    ) -> StyleDistillationTrainer:
        if learning_rate <= 0 or weight_decay < 0 or total_updates <= 0:
            raise ValueError("Invalid style distillation schedule")
        parameters = tuple(model.adapter.parameters()) + tuple(model.policy.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates)
        return cls(
            model,
            optimizer,
            scheduler,
            gradient_clip=gradient_clip,
            beta_kl=beta_kl,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def train_step(self, batch: dict[str, torch.Tensor]) -> StyleDistillationStep:
        self.model.train()
        self.model.base.eval()
        moved = {name: tensor.to(self.device) for name, tensor in batch.items()}
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            base = self.model.base.actor(
                moved["frames"],
                moved["scalars"],
                moved["previous_actions"],
                moved["masks"],
            )
        style_logits = self.model.policy(self.model.adapter(base.features))
        metrics = style_distillation_loss(
            style_logits,
            base.logits,
            moved["actions"],
            moved["valid"],
            moved["present"],
            beta_kl=self.beta_kl,
        )
        metrics.total.backward()
        parameters = tuple(self.model.adapter.parameters()) + tuple(self.model.policy.parameters())
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clip))
        self.optimizer.step()
        self.scheduler.step()
        self.updates += 1
        return StyleDistillationStep(
            total_loss=float(metrics.total.detach()),
            imitation_loss=float(metrics.imitation.detach()),
            base_kl=float(metrics.base_kl.detach()),
            accuracy=metrics.accuracy,
            supervised_count=metrics.supervised_count,
            gradient_norm=gradient_norm,
            update=self.updates,
        )


@torch.no_grad()
def evaluate_style_distillation(
    model: StyledActorCritic,
    batches: Iterable[dict[str, torch.Tensor]],
) -> dict[str, float | int | bool]:
    model.eval()
    device = next(model.parameters()).device
    attack_ids = torch.tensor((9, 10, 11, 12), device=device)
    positive_probability = 0.0
    base_positive_probability = 0.0
    positive_count = 0
    negative_attack_predictions = 0
    negative_count = 0
    for batch in batches:
        moved = {name: tensor.to(device) for name, tensor in batch.items()}
        base = model.base.actor(
            moved["frames"],
            moved["scalars"],
            moved["previous_actions"],
            moved["masks"],
        )
        style_logits = model.policy(model.adapter(base.features))
        style_probability = torch.softmax(style_logits, dim=-1)
        base_probability = torch.softmax(base.logits, dim=-1)
        supervised = moved["valid"]
        positive = supervised & torch.isin(moved["actions"], attack_ids)
        negative = supervised & ~torch.isin(moved["actions"], attack_ids)
        if torch.any(positive):
            positive_probability += float(style_probability[positive][:, attack_ids].sum())
            base_positive_probability += float(base_probability[positive][:, attack_ids].sum())
            positive_count += int(positive.sum())
        if torch.any(negative):
            predictions = style_logits[negative].argmax(dim=-1)
            negative_attack_predictions += int(torch.isin(predictions, attack_ids).sum())
            negative_count += int(negative.sum())
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("Style evaluation requires positive and negative supervision")
    style_positive_rate = positive_probability / positive_count
    base_positive_rate = base_positive_probability / positive_count
    negative_attack_rate = negative_attack_predictions / negative_count
    return {
        "attack_probability_on_positive": style_positive_rate,
        "base_attack_probability_on_positive": base_positive_rate,
        "attack_probability_delta": style_positive_rate - base_positive_rate,
        "negative_attack_prediction_rate": negative_attack_rate,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "passed": style_positive_rate > base_positive_rate and negative_attack_rate < 0.5,
    }


@torch.no_grad()
def evaluate_defensive_distillation(
    model: StyledActorCritic,
    batches: Iterable[dict[str, torch.Tensor]],
) -> dict[str, float | int | bool]:
    model.eval()
    device = next(model.parameters()).device
    objective_task_id = TASK_IDS["objective"]
    target_matches = 0
    base_target_matches = 0
    risk_count = 0
    no_risk_style_idle = 0
    no_risk_base_idle = 0
    no_risk_changed = 0
    no_risk_count = 0
    for batch in batches:
        moved = {name: tensor.to(device) for name, tensor in batch.items()}
        if "task_ids" not in moved:
            raise ValueError("Defensive style evaluation requires task IDs")
        base = model.base.actor(
            moved["frames"],
            moved["scalars"],
            moved["previous_actions"],
            moved["masks"],
        )
        style_logits = model.policy(model.adapter(base.features))
        style_actions = style_logits.argmax(dim=-1)
        base_actions = base.logits.argmax(dim=-1)
        supervised = moved["valid"]
        risk = supervised & (moved["task_ids"] != objective_task_id)
        no_risk = supervised & (moved["task_ids"] == objective_task_id)
        if torch.any(risk):
            target_matches += int((style_actions[risk] == moved["actions"][risk]).sum())
            base_target_matches += int((base_actions[risk] == moved["actions"][risk]).sum())
            risk_count += int(risk.sum())
        if torch.any(no_risk):
            no_risk_style_idle += int((style_actions[no_risk] == int(MacroAction.IDLE)).sum())
            no_risk_base_idle += int((base_actions[no_risk] == int(MacroAction.IDLE)).sum())
            no_risk_changed += int((style_actions[no_risk] != base_actions[no_risk]).sum())
            no_risk_count += int(no_risk.sum())
    if risk_count <= 0 or no_risk_count <= 0:
        raise ValueError("Defensive evaluation requires risk and no-risk supervision")
    target_agreement = target_matches / risk_count
    base_target_agreement = base_target_matches / risk_count
    agreement_delta = target_agreement - base_target_agreement
    style_idle_rate = no_risk_style_idle / no_risk_count
    base_idle_rate = no_risk_base_idle / no_risk_count
    idle_rate_delta = style_idle_rate - base_idle_rate
    return {
        "risk_target_agreement": target_agreement,
        "base_risk_target_agreement": base_target_agreement,
        "risk_target_agreement_delta": agreement_delta,
        "risk_count": risk_count,
        "no_risk_idle_prediction_rate": style_idle_rate,
        "base_no_risk_idle_prediction_rate": base_idle_rate,
        "no_risk_idle_prediction_rate_delta": idle_rate_delta,
        "no_risk_action_drift_rate": no_risk_changed / no_risk_count,
        "no_risk_count": no_risk_count,
        "passed": agreement_delta >= 0.20 and idle_rate_delta <= 0.05,
    }


@torch.no_grad()
def evaluate_explorer_distillation(
    model: StyledActorCritic,
    batches: Iterable[dict[str, torch.Tensor]],
) -> dict[str, float | int | bool]:
    model.eval()
    device = next(model.parameters()).device
    target_matches = 0
    base_target_matches = 0
    route_count = 0
    context_changed = 0
    context_count = 0
    for batch in batches:
        moved = {name: tensor.to(device) for name, tensor in batch.items()}
        base = model.base.actor(
            moved["frames"],
            moved["scalars"],
            moved["previous_actions"],
            moved["masks"],
        )
        style_logits = model.policy(model.adapter(base.features))
        style_actions = style_logits.argmax(dim=-1)
        base_actions = base.logits.argmax(dim=-1)
        supervised = moved["valid"]
        route = supervised & (moved["actions"] != base_actions)
        context = supervised & (moved["actions"] == base_actions)
        if torch.any(route):
            target_matches += int(
                (style_actions[route] == moved["actions"][route]).sum()
            )
            base_target_matches += int(
                (base_actions[route] == moved["actions"][route]).sum()
            )
            route_count += int(route.sum())
        if torch.any(context):
            context_changed += int((style_actions[context] != base_actions[context]).sum())
            context_count += int(context.sum())
    if route_count <= 0 or context_count <= 0:
        raise ValueError("Explorer evaluation requires route and Base context labels")
    target_agreement = target_matches / route_count
    base_target_agreement = base_target_matches / route_count
    agreement_delta = target_agreement - base_target_agreement
    context_drift = context_changed / context_count
    return {
        "route_target_agreement": target_agreement,
        "base_route_target_agreement": base_target_agreement,
        "route_target_agreement_delta": agreement_delta,
        "route_count": route_count,
        "base_context_action_drift_rate": context_drift,
        "base_context_count": context_count,
        "passed": agreement_delta >= 0.20 and context_drift <= 0.10,
    }


def save_style_distillation_checkpoint(
    path: Path,
    *,
    model: StyledActorCritic,
    base_checkpoint_sha256: str,
    scenario_hash: str,
    data_manifest_sha256: str,
    config_hash: str,
    updates: int,
    style: str = "aggressive",
) -> Path:
    if updates <= 0:
        raise ValueError("Style distillation checkpoint requires completed updates")
    payload = {
        "schema_version": 1,
        "kind": "style_distillation",
        "style": style,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "scenario_hash": scenario_hash,
        "data_manifest_sha256": data_manifest_sha256,
        "config_hash": config_hash,
        "updates": updates,
        "model": model.state_dict(),
    }
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return path


def save_style_neutral_checkpoint(
    path: Path,
    *,
    model: StyledActorCritic,
    style: str,
    base_checkpoint_sha256: str,
    scenario_hash: str,
    data_manifest_sha256: str,
    config_hash: str,
) -> Path:
    payload = {
        "schema_version": 1,
        "kind": "style_neutral",
        "style": style,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "scenario_hash": scenario_hash,
        "data_manifest_sha256": data_manifest_sha256,
        "config_hash": config_hash,
        "updates": 0,
        "model": model.state_dict(),
    }
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return path


def load_style_distillation_checkpoint(
    path: Path,
    *,
    model: StyledActorCritic,
    expected_base_checkpoint_sha256: str,
    expected_scenario_hash: str,
    expected_style: str = "aggressive",
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "schema_version": 1,
        "kind": "style_distillation",
        "style": expected_style,
        "base_checkpoint_sha256": expected_base_checkpoint_sha256,
        "scenario_hash": expected_scenario_hash,
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise ValueError("Style distillation checkpoint identity does not match")
    before = model.state_dict()
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("Style distillation checkpoint has no model state")
    base_keys = tuple(name for name in before if name.startswith("base."))
    if any(name not in state or not torch.equal(before[name], state[name]) for name in base_keys):
        raise ValueError("Style distillation checkpoint changed the frozen base")
    model.load_state_dict(state, strict=True)
    return {
        name: payload[name]
        for name in (
            "base_checkpoint_sha256",
            "config_hash",
            "data_manifest_sha256",
            "scenario_hash",
            "updates",
        )
    }
