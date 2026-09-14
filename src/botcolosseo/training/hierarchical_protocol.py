"""Pure contracts shared by hierarchical collection, training and cross-play.

No privileged environment state belongs in this module's control interfaces.
"""

import math
from dataclasses import dataclass
from enum import IntEnum

COMMAND_SCHEMA = "search-bands160-engage-disengage-extract-v2"


class Command(IntEnum):
    SEARCH_NORTH = 0
    SEARCH_CENTER = 1
    SEARCH_SOUTH = 2
    ENGAGE = 3
    DISENGAGE = 4
    EXTRACT_NORTH = 5
    EXTRACT_SOUTH = 6


def search_band(y: float) -> Command:
    """Fixed geometry bands: central anchors at y=0/96 belong to center."""
    if not math.isfinite(y):
        raise ValueError("Coordinate must be finite")
    return (
        Command.SEARCH_NORTH
        if y > 160
        else (Command.SEARCH_SOUTH if y < -160 else Command.SEARCH_CENTER)
    )


@dataclass(frozen=True)
class ControlCondition:
    aggressive: float = 0.0
    defensive: float = 0.0
    explorer: float = 0.0
    difficulty: float = 1.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in self.as_tuple()):
            raise ValueError("Control conditions must be finite and in [0, 1]")

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.aggressive, self.defensive, self.explorer, self.difficulty


def matrix_condition(identity: dict) -> ControlCondition:
    """Legacy empirical games were always Neutral/Hard, never unspecified."""
    values = identity.get("condition", (0.0, 0.0, 0.0, 1.0))
    if not isinstance(values, (tuple, list)) or len(values) != 4:
        raise ValueError("Matrix condition requires A/D/E/difficulty")
    return ControlCondition(*values)


def require_neutral_matrix(identity: dict) -> None:
    if matrix_condition(identity) != ControlCondition():
        raise ValueError("This consumer requires a Neutral/Hard matrix")


def should_replan(elapsed: int, *, public_event: bool = False) -> bool:
    """Called before a low-level action; episode termination is handled separately."""
    if elapsed < 0:
        raise ValueError("elapsed must be nonnegative")
    return elapsed >= 8 or (public_event and elapsed >= 2)


def smdp_target(
    rewards: tuple[float, ...], gamma: float, next_value: float, *, terminated: bool
) -> float:
    """Truncations bootstrap; true terminals do not. Rewards are low-step rewards."""
    if not rewards or not 0 <= gamma <= 1:
        raise ValueError("Need a nonempty transition and gamma in [0, 1]")
    if not all(math.isfinite(x) for x in (*rewards, next_value)):
        raise ValueError("Rewards and values must be finite")
    result = sum(gamma**index * reward for index, reward in enumerate(rewards))
    return result if terminated else result + gamma ** len(rewards) * next_value


def own_payoffs(
    host_banked: float, opponent_banked: float, *, globally_settled: bool
) -> tuple[float, float]:
    """General-sum terminal payoff, never relative score or denial reward."""
    if not globally_settled:
        raise ValueError("Both players must finish before recording payoff")
    values = (host_banked, opponent_banked)
    if not all(math.isfinite(x) and 0 <= x <= 150 for x in values):
        raise ValueError("Banked value exceeds the single-extraction rule contract")
    return host_banked / 150, opponent_banked / 150


@dataclass(frozen=True)
class GameIdentity:
    scenario_hash: str
    rules_hash: str
    executor_hash: str
    host_condition: ControlCondition
    opponent_condition: ControlCondition
    command_schema: str = COMMAND_SCHEMA
    inference: str = "high-sample_low-argmax"
    reward_version: str = "own-banked-div150-v1"

    def __post_init__(self) -> None:
        if not all((self.scenario_hash, self.rules_hash, self.executor_hash)):
            raise ValueError("Game identity requires scenario, rules and executor hashes")

    def require_same_game(self, other: "GameIdentity") -> None:
        if self != other:
            raise ValueError("Stale or incompatible payoff game identity")
