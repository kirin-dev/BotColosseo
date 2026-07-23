from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_types import DuelActorObservation

DIFFICULTIES = ("easy", "normal", "hard")


class PublicPolicy(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: DuelActorObservation) -> MacroAction: ...


@dataclass(frozen=True)
class DifficultyProfile:
    reaction_delay: int
    policy_update_interval: int

    def __post_init__(self) -> None:
        if type(self.reaction_delay) is not int or self.reaction_delay < 0:
            raise ValueError("Difficulty reaction delay must be a nonnegative integer")
        if (
            type(self.policy_update_interval) is not int
            or self.policy_update_interval <= 0
        ):
            raise ValueError("Difficulty policy update interval must be positive")


def load_difficulty_profiles(path: Path) -> dict[str, DifficultyProfile]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "profiles"}
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("profiles"), dict)
        or set(payload["profiles"]) != set(DIFFICULTIES)
    ):
        raise ValueError("Difficulty config does not match schema version 1")
    profiles: dict[str, DifficultyProfile] = {}
    for name in DIFFICULTIES:
        values = payload["profiles"][name]
        if not isinstance(values, dict) or set(values) != {
            "reaction_delay",
            "policy_update_interval",
        }:
            raise ValueError(f"Difficulty profile {name} has invalid fields")
        profiles[name] = DifficultyProfile(**values)
    hard = profiles["hard"]
    if hard != DifficultyProfile(reaction_delay=0, policy_update_interval=1):
        raise ValueError("Hard difficulty must be the native checkpoint policy")
    delays = tuple(profiles[name].reaction_delay for name in DIFFICULTIES)
    intervals = tuple(profiles[name].policy_update_interval for name in DIFFICULTIES)
    if delays != tuple(sorted(delays, reverse=True)) or intervals != tuple(
        sorted(intervals, reverse=True)
    ):
        raise ValueError("Difficulty restrictions must be ordered Easy to Hard")
    return profiles


class DifficultyPolicy:
    def __init__(self, policy: PublicPolicy, profile: DifficultyProfile) -> None:
        self._policy = policy
        self.profile = profile
        self._delay: deque[MacroAction] = deque()
        self._held_action = MacroAction.IDLE
        self._decision = 0
        self._ready = False

    def reset(self) -> None:
        self._policy.reset()
        self._delay = deque(
            [MacroAction.IDLE] * self.profile.reaction_delay,
            maxlen=self.profile.reaction_delay + 1,
        )
        self._held_action = MacroAction.IDLE
        self._decision = 0
        self._ready = True

    def act(self, observation: DuelActorObservation) -> MacroAction:
        if not self._ready:
            raise RuntimeError("Difficulty policy must be reset before act")
        if self._decision % self.profile.policy_update_interval == 0:
            try:
                self._held_action = MacroAction(self._policy.act(observation))
            except (TypeError, ValueError) as error:
                raise ValueError("Wrapped policy returned an invalid action") from error
        self._decision += 1
        self._delay.append(self._held_action)
        return self._delay.popleft()
