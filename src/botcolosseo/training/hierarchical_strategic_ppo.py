"""Single-episode strategic PPO update; no executor parameters are owned here."""

import math

import torch

from botcolosseo.agents.hierarchical_critic import StrategicActorCritic
from botcolosseo.training.hierarchical_smdp import strategic_gae
from botcolosseo.training.ppo import clipped_ppo_loss

HIGH_INPUT_KEYS = (
    "features",
    "scalars",
    "previous_commands",
    "elapsed",
    "style",
    "difficulty",
    "masks",
)


def update_strategic_ppo(
    model: StrategicActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    gamma: float = 0.997,
    gae_lambda: float = 0.95,
    chunk_size: int = 64,
    reference=None,
    reference_weight: float = 0.0,
) -> dict[str, float]:
    """Task PPO with optional fixed conditional-reference retention.

    Requires a whole local episode beginning at reset. Stored executor features
    are constants; high recurrent memory is recomputed with current parameters.
    """
    if chunk_size <= 0 or batch["masks"].shape[0] != 1 or batch["masks"][0, 0] != 0:
        raise ValueError("Need a full reset-started single episode")
    if not math.isfinite(reference_weight) or reference_weight < 0:
        raise ValueError("Reference weight must be finite and nonnegative")
    if reference_weight and reference is None:
        raise ValueError("Positive retention weight requires a reference")
    if reference is not None:
        if any(p.requires_grad for p in reference.parameters()):
            raise ValueError("Strategic reference must be frozen")
        reference.eval()
    owned = {id(p) for p in model.parameters()}
    if any(id(p) not in owned for g in optimizer.param_groups for p in g["params"]):
        raise ValueError("Optimizer may contain only strategic model parameters")
    with torch.no_grad():
        gae = strategic_gae(
            batch["rewards"],
            batch["old_values"],
            batch["next_values"],
            batch["durations"],
            batch["terminated"],
            batch["truncated"],
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        advantages = gae.advantages
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(
            1e-8
        )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    length = batch["rewards"].shape[1]
    hidden = reference_hidden = None
    total = kl = reference_kl = 0.0
    for start in range(0, length, chunk_size):
        part = {k: v[:, start : start + chunk_size].detach() for k, v in batch.items()}
        output = model(
            privileged=part["privileged"], hidden=hidden, **{k: part[k] for k in HIGH_INPUT_KEYS}
        )
        distribution = torch.distributions.Categorical(logits=output.logits)
        loss = clipped_ppo_loss(
            new_log_probs=distribution.log_prob(part["commands"]),
            entropy=distribution.entropy(),
            values=output.values,
            old_log_probs=part["old_log_probs"],
            old_values=part["old_values"],
            advantages=advantages[:, start : start + chunk_size],
            returns=gae.returns[:, start : start + chunk_size],
            valid=torch.ones_like(output.values, dtype=torch.bool),
            policy_clip=0.1,
            value_clip=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.001,
            max_kl=0.03,
        )
        weight = output.values.shape[1] / length
        retention = output.logits.new_zeros(())
        if reference is not None and reference_weight:
            with torch.no_grad():
                anchor = reference(hidden=reference_hidden, **{k: part[k] for k in HIGH_INPUT_KEYS})
                reference_hidden = anchor.hidden
            retention = torch.distributions.kl_divergence(
                torch.distributions.Categorical(logits=anchor.logits), distribution
            ).mean()
        objective = loss.total_loss + reference_weight * retention
        (weight * objective).backward()
        total += weight * float(objective.detach())
        reference_kl += weight * float(retention.detach())
        kl += weight * loss.approximate_kl
        hidden = output.hidden.detach()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {
        "loss": total,
        "approximate_kl": kl,
        "reference_kl": reference_kl,
        "gradient_norm": float(gradient),
    }
