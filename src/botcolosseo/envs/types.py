from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from botcolosseo.envs.actions import MacroAction

if TYPE_CHECKING:
    from botcolosseo.envs.events import EpisodeEvent


@dataclass(frozen=True)
class ActorObservation:
    frame: NDArray[np.uint8]
    health: float
    ammo: float
    attack_ready: bool
    has_core: bool
    home_score: int
    away_score: int
    remaining_tics: int
    previous_action: MacroAction

    def __post_init__(self) -> None:
        frame = np.asarray(self.frame)
        if frame.shape != (84, 84) or frame.dtype != np.uint8:
            raise ValueError(
                f"Actor frame must be 84x84 uint8 grayscale, got {frame.shape} {frame.dtype}"
            )
        frozen_frame = np.ascontiguousarray(frame).copy()
        frozen_frame.setflags(write=False)
        object.__setattr__(self, "frame", frozen_frame)


@dataclass(frozen=True)
class PrivilegedState:
    player_x: float
    player_y: float
    player_angle: float
    region_name: str | None
    core_x: float
    core_y: float
    target_x: float
    target_y: float
    target_alive: bool
    task_phase: int
    has_core: bool


@dataclass(frozen=True)
class TaskStep:
    observation: ActorObservation
    reward: float
    terminated: bool
    truncated: bool
    events: tuple[EpisodeEvent, ...]
