from __future__ import annotations

from typing import NamedTuple

import torch


class GAEOutput(NamedTuple):
    advantages: torch.Tensor
    returns: torch.Tensor


def _require_finite(*tensors: torch.Tensor) -> None:
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise FloatingPointError("GAE inputs must be finite")


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> GAEOutput:
    if rewards.shape != values.shape or rewards.shape != next_values.shape:
        raise ValueError("GAE reward and value tensors must share a shape")
    if rewards.ndim != 2 or rewards.shape[1] == 0:
        raise ValueError("GAE tensors must have shape [environments, time]")
    if terminated.shape != rewards.shape or truncated.shape != rewards.shape:
        raise ValueError("GAE boundary tensors have the wrong shape")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError("GAE boundaries must be boolean")
    if bool((terminated & truncated).any()):
        raise ValueError("A transition cannot be terminal and truncated")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    _require_finite(rewards, values, next_values)

    bootstrap = (~terminated).to(values.dtype)
    delta = rewards + gamma * next_values * bootstrap - values
    continuation = (~(terminated | truncated)).to(values.dtype)
    advantages = torch.zeros_like(values)
    accumulator = torch.zeros_like(values[:, 0])
    for index in range(values.shape[1] - 1, -1, -1):
        accumulator = (
            delta[:, index]
            + gamma * gae_lambda * continuation[:, index] * accumulator
        )
        advantages[:, index] = accumulator
    returns = advantages + values
    _require_finite(advantages, returns)
    return GAEOutput(advantages, returns)


def normalize_advantages(
    advantages: torch.Tensor,
    valid: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    if advantages.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError("Advantage values and valid mask are incompatible")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    _require_finite(advantages)
    selected = advantages[valid]
    if selected.numel() == 0:
        raise ValueError("Cannot normalize an empty advantage set")
    normalized = torch.zeros_like(advantages)
    normalized[valid] = (selected - selected.mean()) / (
        selected.std(unbiased=False) + epsilon
    )
    return normalized
