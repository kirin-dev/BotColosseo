"""Command-balanced recurrent BC with episode-local truncated backpropagation."""

from __future__ import annotations

import math

import torch
from torch import nn

from botcolosseo.agents.hierarchical_model import CommandExecutor


def fit_counterfactual_episode(
    actor: CommandExecutor,
    batch: dict[str, torch.Tensor],
    labels: torch.Tensor,
    valid: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    stride: int = 16,
) -> float:
    """Train one-step alternatives from the same recorded recurrent history.

    Never carry an alternative command's hidden state into the factual trajectory.
    All valid labels use ordinary CE, not an artificial action-separation penalty.
    """
    if stride <= 0 or labels.shape != (*batch["commands"].shape, 7) or valid.shape != labels.shape:
        raise ValueError("Invalid counterfactual sequence")
    selected = valid[:, ::stride].sum()
    if selected <= 0:
        raise ValueError("No applicable counterfactual labels")
    optimizer.zero_grad(set_to_none=True)
    actor.train()
    hidden = None
    total = 0.0
    keys = ("frames", "scalars", "previous_actions", "masks", "commands", "difficulty")
    if batch["commands"].shape[0] != 1:
        raise ValueError("Counterfactual updater expects one episode")
    for t in range(batch["commands"].shape[1]):
        factual = {k: batch[k][:, t : t + 1] for k in keys}
        if t % stride == 0 and valid[0, t].any():
            alternatives = {k: v.expand(7, *v.shape[1:]) for k, v in factual.items()}
            alternatives["commands"] = torch.arange(7, device=labels.device)[:, None]
            out = actor(**alternatives, hidden=None if hidden is None else hidden.expand(1, 7, -1))
            ce = nn.functional.cross_entropy(out.logits[:, 0], labels[0, t], reduction="none")
            loss = (ce * valid[0, t]).sum() / selected
            loss.backward()
            total += float(loss.detach())
        with torch.no_grad():
            hidden = actor(**factual, hidden=hidden).hidden
    nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return total


@torch.no_grad()
def evaluate_episode(
    actor: CommandExecutor, batch: dict[str, torch.Tensor], *, chunk_size: int = 64
) -> dict[str, torch.Tensor]:
    """Accumulate per-command sufficient statistics, without resetting at switches."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    actor.eval()
    device = batch["frames"].device
    counts = torch.zeros(7, device=device)
    loss_sum = torch.zeros(7, device=device)
    correct = torch.zeros(7, device=device)
    switch_counts = torch.zeros(7, device=device)
    switch_correct = torch.zeros(7, device=device)
    switches = torch.zeros_like(batch["valid"], dtype=torch.bool)
    switches[:, 1:] = batch["commands"][:, 1:] != batch["commands"][:, :-1]
    hidden = None
    for start in range(0, batch["valid"].shape[1], chunk_size):
        part = {key: value[:, start : start + chunk_size] for key, value in batch.items()}
        out = actor(
            part["frames"],
            part["scalars"],
            part["previous_actions"],
            part["masks"],
            part["commands"],
            part["difficulty"],
            hidden,
        )
        loss = nn.functional.cross_entropy(
            out.logits.flatten(0, 1), part["actions"].flatten(), reduction="none"
        )
        commands = part["commands"].flatten()
        valid = part["valid"].flatten().float()
        counts.scatter_add_(0, commands, valid)
        loss_sum.scatter_add_(0, commands, loss * valid)
        hits = (out.logits.argmax(-1) == part["actions"]).flatten().float()
        correct.scatter_add_(0, commands, hits * valid)
        switched_valid = switches[:, start : start + chunk_size].flatten() * valid
        switch_counts.scatter_add_(0, commands, switched_valid)
        switch_correct.scatter_add_(0, commands, hits * switched_valid)
        hidden = out.hidden
    return {
        "counts": counts,
        "loss_sum": loss_sum,
        "correct": correct,
        "switch_counts": switch_counts,
        "switch_correct": switch_correct,
    }


def command_weights(counts: torch.Tensor) -> torch.Tensor:
    if counts.shape != (7,) or not bool(torch.isfinite(counts).all() and (counts > 0).all()):
        raise ValueError("Every command needs valid demonstrations before BC")
    weights = counts.float().reciprocal()
    return weights / weights.mean()


def fit_episode(
    actor: CommandExecutor,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    weights: torch.Tensor,
    *,
    chunk_size: int = 64,
    normalization_mass: float | None = None,
) -> dict[str, float]:
    """One update per episode; no optimizer mutation while carrying its hidden state.

    Invalid waiting frames may advance memory but never contribute action loss.
    TBPTT detaches only at chunk boundaries, never at command switches.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    valid = batch["valid"].bool()
    denominator = (weights[batch["commands"]] * valid).sum()
    if denominator <= 0:
        raise ValueError("Episode contains no valid labels")
    if normalization_mass is not None:
        if not math.isfinite(normalization_mass) or normalization_mass <= 0:
            raise ValueError("Normalization mass must be finite and positive")
        # A dataset-wide mean mass preserves global command weighting under
        # uniform episode sampling; per-episode division cancels single-command weights.
        denominator = normalization_mass
    optimizer.zero_grad(set_to_none=True)
    actor.train()
    hidden = None
    total_loss = 0.0
    correct = 0
    for start in range(0, valid.shape[1], chunk_size):
        stop = start + chunk_size
        part = {key: value[:, start:stop] for key, value in batch.items()}
        out = actor(
            part["frames"],
            part["scalars"],
            part["previous_actions"],
            part["masks"],
            part["commands"],
            part["difficulty"],
            hidden,
        )
        losses = nn.functional.cross_entropy(
            out.logits.flatten(0, 1), part["actions"].flatten(), reduction="none"
        ).reshape_as(part["actions"])
        weighted = (losses * weights[part["commands"]] * part["valid"]).sum() / denominator
        weighted.backward()
        total_loss += float(weighted.detach())
        correct += int(((out.logits.argmax(-1) == part["actions"]) & part["valid"]).sum())
        hidden = out.hidden.detach()
    gradient = nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {
        "loss": total_loss,
        "accuracy": correct / int(valid.sum()),
        "gradient_norm": float(gradient),
    }
