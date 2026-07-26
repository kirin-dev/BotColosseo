from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_protocol import ExtractionEvent
from botcolosseo.envs.extraction_rules import EXTRACTION_REQUIRED_TICS, LifeState


@dataclass(frozen=True)
class ExtractionActorObservation:
    frame: NDArray[np.uint8]
    health: float
    ammo: float
    carried_value: int
    free_slots: int
    minimum_slot_value: int
    banked_value: int
    extraction_open: bool
    extraction_progress: float
    remaining_time: float
    previous_action: int

    def __post_init__(self) -> None:
        frame = np.asarray(self.frame)
        if frame.shape != (84, 84) or frame.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 extraction frame [84, 84], "
                f"got {frame.shape}/{frame.dtype}"
            )
        frozen = np.array(frame, dtype=np.uint8, copy=True)
        frozen.flags.writeable = False
        object.__setattr__(self, "frame", frozen)
        if not 0.0 <= self.health <= 100.0:
            raise ValueError(f"Invalid extraction health: {self.health}")
        if not 0.0 <= self.ammo <= 40.0:
            raise ValueError(f"Invalid extraction ammo: {self.ammo}")
        if self.carried_value < 0 or self.banked_value < 0:
            raise ValueError("Extraction values must be nonnegative")
        if not 0 <= self.free_slots <= 3:
            raise ValueError("Invalid extraction free-slot count")
        if self.minimum_slot_value not in (0, 10, 25, 50):
            raise ValueError("Invalid extraction minimum slot value")
        if not 0.0 <= self.extraction_progress <= 1.0:
            raise ValueError("Invalid normalized extraction progress")
        if not 0.0 <= self.remaining_time <= 75.0:
            raise ValueError("Invalid extraction remaining time")
        MacroAction(self.previous_action)


@dataclass(frozen=True)
class ExtractionPrivilegedState:
    host_x: float
    host_y: float
    host_angle: float
    opponent_x: float
    opponent_y: float
    opponent_angle: float
    host_health: float
    opponent_health: float
    host_slots: tuple[int, int, int]
    opponent_slots: tuple[int, int, int]
    host_banked: int
    opponent_banked: int
    cache_owner: int
    cache_slots: tuple[int, int, int]
    cache_x: float
    cache_y: float
    world_loot_mask: int
    round_state: int
    winner: int
    engine_tic: int


@dataclass(frozen=True)
class ExtractionObservations:
    host: ExtractionActorObservation
    opponent: ExtractionActorObservation


@dataclass(frozen=True)
class ExtractionResetInfo:
    seed: int
    port: int
    episode_id: int
    engine_tic: int
    protocol_version: int
    scenario_hash: str


@dataclass(frozen=True)
class ExtractionStep:
    host: ExtractionActorObservation
    opponent: ExtractionActorObservation
    host_reward: float
    opponent_reward: float
    terminated: bool
    truncated: bool
    winner: int
    events: tuple[ExtractionEvent, ...]
    decision_index: int
    engine_tic: int
    peer_tic_lag: int


def observation_health(life_state: int, native_health: float) -> float:
    if life_state != int(LifeState.ACTIVE):
        return 0.0
    return max(0.0, min(float(native_health), 100.0))


def normalized_extraction_progress(progress_tics: int) -> float:
    return progress_tics / EXTRACTION_REQUIRED_TICS
