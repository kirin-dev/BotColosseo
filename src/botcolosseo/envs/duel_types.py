from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_protocol import DuelEvent


@dataclass(frozen=True)
class DuelActorObservation:
    frame: NDArray[np.uint8]
    health: float
    armor: float
    ammo: float
    own_score: int
    opponent_score: int
    has_core: bool
    previous_action: int

    def __post_init__(self) -> None:
        frame = np.asarray(self.frame)
        if frame.shape != (84, 84) or frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 frame [84, 84], got {frame.shape}/{frame.dtype}")
        frozen = np.array(frame, dtype=np.uint8, copy=True)
        frozen.flags.writeable = False
        object.__setattr__(self, "frame", frozen)
        if not 0.0 <= self.health <= 200.0:
            raise ValueError(f"Invalid health: {self.health}")
        if not 0.0 <= self.armor <= 200.0 or self.ammo < 0.0:
            raise ValueError("Armor and ammo must be nonnegative and bounded")
        if self.own_score < 0 or self.opponent_score < 0:
            raise ValueError("Scores must be nonnegative")
        MacroAction(self.previous_action)


@dataclass(frozen=True)
class DuelPrivilegedState:
    host_x: float
    host_y: float
    host_angle: float
    host_region: str | None
    opponent_x: float
    opponent_y: float
    opponent_angle: float
    opponent_region: str | None
    core_x: float
    core_y: float
    carrier: int
    host_health: float
    opponent_health: float
    host_score: int
    opponent_score: int
    round_state: int
    engine_tic: int


@dataclass(frozen=True)
class DuelStep:
    host: DuelActorObservation
    opponent: DuelActorObservation
    host_reward: float
    opponent_reward: float
    terminated: bool
    truncated: bool
    events: tuple[DuelEvent, ...]
    decision_index: int
    engine_tic: int
