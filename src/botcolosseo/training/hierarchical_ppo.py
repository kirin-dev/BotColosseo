"""Conservative command PPO using frozen-reference KL and demonstration replay."""

from __future__ import annotations

import torch

from botcolosseo.agents.hierarchical_model import CommandActorCritic, CommandExecutor
from botcolosseo.training.gae import generalized_advantage_estimate
from botcolosseo.training.ppo import clipped_ppo_loss


def update_command_ppo(
    model: CommandActorCritic,
    reference: CommandExecutor,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, torch.Tensor],
    replay: dict[str, torch.Tensor],
    *,
    gamma: float = 0.99,
    chunk_size: int = 64,
    reference_kl: float = 0.1,
    replay_weight: float = 0.1,
) -> dict[str, float]:
    if chunk_size <= 0 or reference_kl < 0 or replay_weight < 0:
        raise ValueError("Invalid PPO settings")
    if any(p.requires_grad for p in reference.parameters()):
        raise ValueError("Reference must be frozen")
    reference.eval()
    model.train()
    values = rollout["old_values"]
    next_values = torch.cat((values[:, 1:], rollout["bootstrap"]), 1)
    truncated = torch.zeros_like(rollout["terminal"])
    truncated[:, -1] = ~rollout["terminal"][:, -1]
    gae = generalized_advantage_estimate(
        rollout["rewards"],
        values,
        next_values,
        rollout["terminal"],
        truncated,
        gamma=gamma,
        gae_lambda=0.95,
    )
    advantages = gae.advantages
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    optimizer.zero_grad(set_to_none=True)
    hidden = reference_hidden = None
    total = kl_total = 0.0
    keys = ("frames", "scalars", "previous_actions", "masks", "commands", "difficulty")
    length = values.shape[1]
    for start in range(0, length, chunk_size):
        stop = start + chunk_size
        inputs = {k: rollout[k][:, start:stop] for k in keys}
        output = model(**inputs, privileged=rollout["privileged"][:, start:stop], hidden=hidden)
        with torch.no_grad():
            anchor = reference(**inputs, hidden=reference_hidden)
        distribution = torch.distributions.Categorical(logits=output.logits)
        old_slice = slice(start, stop)
        loss = clipped_ppo_loss(
            new_log_probs=distribution.log_prob(rollout["actions"][:, old_slice]),
            entropy=distribution.entropy(),
            values=output.values,
            old_log_probs=rollout["old_log_probs"][:, old_slice],
            old_values=values[:, old_slice],
            advantages=advantages[:, old_slice],
            returns=gae.returns[:, old_slice],
            valid=torch.ones_like(output.values, dtype=torch.bool),
            policy_clip=0.1,
            value_clip=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.001,
            max_kl=0.03,
        )
        kl = torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=anchor.logits), distribution
        ).mean()
        weight = output.values.shape[1] / length
        objective = weight * (loss.total_loss + reference_kl * kl)
        objective.backward()
        total += float(objective.detach())
        kl_total += weight * float(kl.detach())
        hidden = output.hidden.detach()
        reference_hidden = anchor.hidden
    # Preserve whole-episode replay memory; no updates between recurrent chunks.
    hidden = None
    denominator = replay["valid"].sum()
    if denominator <= 0:
        raise ValueError("Replay episode needs valid labels")
    for start in range(0, replay["valid"].shape[1], chunk_size):
        part = {key: value[:, start : start + chunk_size] for key, value in replay.items()}
        output = model.actor(**{k: part[k] for k in keys}, hidden=hidden)
        ce = torch.nn.functional.cross_entropy(
            output.logits.flatten(0, 1), part["actions"].flatten(), reduction="none"
        ).reshape_as(part["actions"])
        objective = replay_weight * (ce * part["valid"]).sum() / denominator
        objective.backward()
        total += float(objective.detach())
        hidden = output.hidden.detach()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {"loss": total, "reference_kl": kl_total, "gradient_norm": float(gradient)}
