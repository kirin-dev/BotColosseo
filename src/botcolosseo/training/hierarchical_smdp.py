"""Variable-duration high-level returns from one player's low-step trajectory."""

import math
from dataclasses import dataclass

import torch

from botcolosseo.training.gae import GAEOutput
from botcolosseo.training.hierarchical_protocol import ControlCondition


@dataclass(frozen=True)
class StrategicTransition:
    start: int
    duration: int
    command: int
    log_prob: float
    condition: ControlCondition
    reward: float
    bootstrap_discount: float
    terminated: bool
    truncated: bool


def strategic_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    durations: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> GAEOutput:
    """GAE with gamma**duration and lambda per strategic transition.

    Rewards are already discounted within each segment. Episode boundaries stop
    traces; truncation retains the value bootstrap for the final observation.
    """
    tensors = (rewards, values, next_values, durations, terminated, truncated)
    if rewards.ndim != 2 or rewards.shape[1] == 0 or any(t.shape != rewards.shape for t in tensors):
        raise ValueError("Expected aligned [batch, high_time] tensors")
    if any(t.device != rewards.device for t in tensors):
        raise ValueError("SMDP tensors must share a device")
    if durations.dtype != torch.long or not bool((durations > 0).all()):
        raise ValueError("Durations must be positive long integers")
    if (
        terminated.dtype != torch.bool
        or truncated.dtype != torch.bool
        or bool((terminated & truncated).any())
    ):
        raise ValueError("Invalid terminal/truncation flags")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("Discount and lambda must be in [0,1]")
    if any(not t.is_floating_point() or not bool(torch.isfinite(t).all()) for t in tensors[:3]):
        raise ValueError("Reward/value tensors must be finite floating point")
    discounts = gamma ** durations.to(values.dtype)
    delta = rewards + discounts * (~terminated) * next_values - values
    advantages = torch.zeros_like(values)
    carry = torch.zeros_like(values[:, 0])
    for index in range(values.shape[1] - 1, -1, -1):
        carry = (
            delta[:, index]
            + discounts[:, index] * gae_lambda * (~(terminated | truncated))[:, index] * carry
        )
        advantages[:, index] = carry
    return GAEOutput(advantages, advantages + values)


def aggregate_strategic_transitions(
    records: list[dict],
    rewards: list[float],
    *,
    gamma: float,
    terminated: bool,
    truncated: bool,
) -> list[StrategicTransition]:
    """Require a closed local episode; never append inactive-player waiting steps.

    A local terminal player may precede global payoff settlement. The caller
    continues the opponent separately and must not insert those steps here.
    Rewards must already use the declared high-level objective, not low shaping.
    """
    if not records or len(records) != len(rewards):
        raise ValueError("Need aligned nonempty records and rewards")
    if terminated == truncated or not 0 <= gamma <= 1:
        raise ValueError("Need one final boundary and a valid discount")
    if not all(math.isfinite(r) for r in rewards):
        raise ValueError("Nonfinite rewards")
    if not records[0]["replanned"]:
        raise ValueError("First action requires a high-level decision")
    starts = [i for i, r in enumerate(records) if r["replanned"]]
    result = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(records)
        head = records[start]
        command = head["command"]
        log_prob = head["command_log_prob"]
        if not 0 <= command < 7 or log_prob is None or not math.isfinite(log_prob) or log_prob > 0:
            raise ValueError("Invalid sampled command or log probability")
        for i in range(start, end):
            row = records[i]
            if row["decision"] != i or row["command"] != command:
                raise ValueError("Noncontiguous trajectory or command changed without replan")
            if (
                tuple(row["style"]) != tuple(head["style"])
                or row["high_difficulty"] != head["high_difficulty"]
            ):
                raise ValueError("High condition changed without replan")
        final = end == len(records)
        duration = end - start
        result.append(
            StrategicTransition(
                start=start,
                duration=duration,
                command=command,
                log_prob=log_prob,
                condition=ControlCondition(*head["style"], head["high_difficulty"]),
                reward=sum(gamma**t * r for t, r in enumerate(rewards[start:end])),
                bootstrap_discount=0.0 if final and terminated else gamma**duration,
                terminated=final and terminated,
                truncated=final and truncated,
            )
        )
    return result
