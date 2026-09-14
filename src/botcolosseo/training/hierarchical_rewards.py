"""Explicit own-task reward and potential shaping contracts for command PPO."""

from __future__ import annotations

import math
from dataclasses import dataclass

from botcolosseo.agents.extraction_teachers import (
    opponent_health,
    opponent_position,
    player_pose,
    player_slots,
)
from botcolosseo.envs.extraction_layouts import randomized_loot_layout
from botcolosseo.envs.extraction_types import ExtractionPrivilegedState
from botcolosseo.training.hierarchical_protocol import Command, search_band


def command_potential(
    state: ExtractionPrivilegedState, *, side: str, command: Command, layout_variant: int
) -> float:
    """Training-only bounded geometry potential on state augmented with command.

    This is navigation progress, not a replacement for event-based command success.
    Engaging closes distance; disengaging increases it. Killing or preventing an
    opponent's extraction never directly changes own task payoff.
    """
    command = Command(command)
    position = player_pose(state, side)[:2]
    if command in (Command.ENGAGE, Command.DISENGAGE):
        if opponent_health(state, side) <= 0:
            return 0.0
        distance = min(math.dist(position, opponent_position(state, side)) / 1024, 1.0)
        return -distance if command == Command.ENGAGE else distance
    if command in (Command.EXTRACT_NORTH, Command.EXTRACT_SOUTH):
        target = (0.0, 400.0 if command == Command.EXTRACT_NORTH else -400.0)
        return -min(math.dist(position, target) / 1024, 1.0)
    minimum = min(player_slots(state, side))
    targets = [
        (x, y)
        for index, (value, x, y) in enumerate(randomized_loot_layout(layout_variant))
        if value > minimum and state.world_loot_mask & (1 << index) and search_band(y) == command
    ]
    if not targets:
        return 0.0
    return -min(min(math.dist(position, target) for target in targets) / 1024, 1.0)


@dataclass(frozen=True)
class CommandReward:
    task: float
    potential: float
    command_success: float

    @property
    def total(self) -> float:
        return self.task + self.potential + self.command_success


def command_reward(
    *,
    previous_banked: float,
    banked: float,
    previous_potential: float,
    next_potential: float,
    gamma: float,
    terminated: bool,
    newly_completed: bool,
    success_bonus: float = 0.05,
    task_weight: float = 1.0,
) -> CommandReward:
    """Caller owns one-shot command completion and augmented-state potentials.

    Potentials include the active command, including at command switches. Do not
    silently zero potential on a nonterminal switch. A time-limit truncation is
    not `terminated`; it retains potential and the value bootstrap.
    Additional command success reward is intentionally not claimed to be PBRS.
    No opponent payoff, kill count or denial signal enters task reward.
    """
    values = (
        previous_banked,
        banked,
        previous_potential,
        next_potential,
        gamma,
        success_bonus,
        task_weight,
    )
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Reward inputs must be finite")
    if not 0 <= previous_banked <= banked <= 150:
        raise ValueError("Banked values must respect monotone single-extraction contract")
    if not 0 <= gamma <= 1 or not 0 <= success_bonus <= 0.1:
        raise ValueError("Invalid discount or bounded success bonus")
    if not 0 < task_weight <= 1:
        raise ValueError("Low-level task weight must be in (0,1]")
    if not -1 <= previous_potential <= 1 or not -1 <= next_potential <= 1:
        raise ValueError("Command potential must be bounded in [-1,1]")
    endpoint = 0.0 if terminated else next_potential
    return CommandReward(
        task_weight * (banked - previous_banked) / 150,
        gamma * endpoint - previous_potential,
        success_bonus if newly_completed else 0.0,
    )
