from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum

import vizdoom as vzd

from botcolosseo.envs.extraction_rules import EXTRACTION_REQUIRED_TICS, LOOT_VALUES

PROTOCOL_VERSION = 3
MAX_WORLD_LOOT = 32


class ExtractionEventType(str, Enum):
    LOOT_PICKUP = "loot_pickup"
    LOOT_DROP = "loot_drop"
    CACHE_CREATED = "cache_created"
    CACHE_LOOTED = "cache_looted"
    EXTRACTION_STARTED = "extraction_started"
    EXTRACTION_INTERRUPTED = "extraction_interrupted"
    EXTRACTED = "extracted"
    VALID_HIT = "valid_hit"
    DEATH = "death"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class ExtractionEvent:
    type: ExtractionEventType
    side: str
    count: int
    value: int
    episode_id: int
    decision_index: int
    engine_tic: int


@dataclass(frozen=True)
class ExtractionPublicState:
    life_state: int
    slots: tuple[int, int, int]
    banked_value: int
    extraction_zone: int
    extraction_progress_tics: int
    extraction_open: bool

    @property
    def carried_value(self) -> int:
        return sum(self.slots)

    @property
    def free_slots(self) -> int:
        return sum(value == 0 for value in self.slots)

    @property
    def minimum_slot_value(self) -> int:
        return min((value for value in self.slots if value), default=0)


@dataclass(frozen=True)
class ExtractionProtocolSnapshot:
    protocol_version: int
    engine_tic: int
    round_state: int
    loot_template: int
    extraction_open: int
    host_life_state: int
    opponent_life_state: int
    host_slot_0: int
    host_slot_1: int
    host_slot_2: int
    opponent_slot_0: int
    opponent_slot_1: int
    opponent_slot_2: int
    host_banked: int
    opponent_banked: int
    host_extraction_zone: int
    opponent_extraction_zone: int
    host_extraction_progress: int
    opponent_extraction_progress: int
    host_loot_pickups: int
    opponent_loot_pickups: int
    host_loot_drops: int
    opponent_loot_drops: int
    host_cache_created: int
    opponent_cache_created: int
    host_cache_looted: int
    opponent_cache_looted: int
    host_extraction_started: int
    opponent_extraction_started: int
    host_extraction_interrupted: int
    opponent_extraction_interrupted: int
    host_extracted: int
    opponent_extracted: int
    host_valid_hits: int
    opponent_valid_hits: int
    host_deaths: int
    opponent_deaths: int
    timeouts: int
    winner: int
    cache_owner: int
    cache_slot_0: int
    cache_slot_1: int
    cache_slot_2: int
    cache_x: float
    cache_y: float
    last_loot_value: int
    last_loot_side: int
    last_damage: int
    world_loot_mask: int
    last_loot_id: int
    last_drop_value: int
    event_serial: int
    reserved_zero: int

    @classmethod
    def from_values(
        cls, values: list[int] | tuple[int, ...]
    ) -> ExtractionProtocolSnapshot:
        if len(values) != len(fields(cls)):
            raise ValueError(
                f"Extraction protocol requires {len(fields(cls))} values"
            )
        converted = [int(value) for value in values]
        converted[43] = vzd.doom_fixed_to_float(converted[43])
        converted[44] = vzd.doom_fixed_to_float(converted[44])
        snapshot = cls(*converted)
        snapshot.validate()
        return snapshot

    def to_values(self) -> tuple[int, ...]:
        values = [int(getattr(self, field.name)) for field in fields(self)]
        values[43] = int(round(self.cache_x * 65536.0))
        values[44] = int(round(self.cache_y * 65536.0))
        return tuple(values)

    def public_state(self, side: str) -> ExtractionPublicState:
        if side == "host":
            prefix = "host"
        elif side == "opponent":
            prefix = "opponent"
        else:
            raise ValueError(f"Unsupported extraction side: {side}")
        return ExtractionPublicState(
            life_state=int(getattr(self, f"{prefix}_life_state")),
            slots=tuple(
                int(getattr(self, f"{prefix}_slot_{index}")) for index in range(3)
            ),
            banked_value=int(getattr(self, f"{prefix}_banked")),
            extraction_zone=int(getattr(self, f"{prefix}_extraction_zone")),
            extraction_progress_tics=int(
                getattr(self, f"{prefix}_extraction_progress")
            ),
            extraction_open=bool(self.extraction_open),
        )

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                f"Unsupported extraction protocol: {self.protocol_version}"
            )
        if self.engine_tic < 0 or self.event_serial < 0:
            raise ValueError("Extraction tics and serials must be nonnegative")
        if self.round_state not in range(3):
            raise ValueError(f"Invalid extraction round state: {self.round_state}")
        if self.loot_template not in (0, 1):
            raise ValueError("Invalid extraction loot template")
        if self.extraction_open not in (0, 1):
            raise ValueError("Extraction open flag must be binary")
        if self.host_life_state not in range(4) or self.opponent_life_state not in range(
            4
        ):
            raise ValueError("Invalid extraction life state")
        if self.winner not in range(4) or self.cache_owner not in range(3):
            raise ValueError("Invalid extraction outcome state")
        if self.host_extraction_zone not in range(
            3
        ) or self.opponent_extraction_zone not in range(3):
            raise ValueError("Invalid extraction zone")
        if not 0 <= self.host_extraction_progress <= EXTRACTION_REQUIRED_TICS or not (
            0 <= self.opponent_extraction_progress <= EXTRACTION_REQUIRED_TICS
        ):
            raise ValueError("Invalid extraction progress")
        slots = (
            self.host_slot_0,
            self.host_slot_1,
            self.host_slot_2,
            self.opponent_slot_0,
            self.opponent_slot_1,
            self.opponent_slot_2,
            self.cache_slot_0,
            self.cache_slot_1,
            self.cache_slot_2,
        )
        if any(value != 0 and value not in LOOT_VALUES for value in slots):
            raise ValueError("Invalid extraction loot slot")
        if self.last_loot_value not in (*LOOT_VALUES, 0):
            raise ValueError("Invalid last loot value")
        if self.last_drop_value not in (*LOOT_VALUES, 0):
            raise ValueError("Invalid last drop value")
        if self.last_loot_side not in range(3):
            raise ValueError("Invalid last loot side")
        if self.last_damage not in (0, 20):
            raise ValueError("Invalid fixed damage")
        if not 0 <= self.world_loot_mask < 2**7:
            raise ValueError("Invalid world loot mask")
        if not 0 <= self.last_loot_id <= MAX_WORLD_LOOT:
            raise ValueError("Invalid loot identity")
        if self.reserved_zero != 0:
            raise ValueError("Extraction reserved_zero must remain zero")
        if any(value < 0 for value in self._counter_values()):
            raise ValueError("Extraction counters must be nonnegative")
        if self.host_banked < 0 or self.opponent_banked < 0:
            raise ValueError("Extraction banked values must be nonnegative")

    def _counter_values(self) -> tuple[int, ...]:
        return tuple(getattr(self, name) for name, _, _, _ in _EVENT_COUNTERS)


_EVENT_COUNTERS = (
    ("host_loot_pickups", ExtractionEventType.LOOT_PICKUP, "host", "last_loot_value"),
    (
        "opponent_loot_pickups",
        ExtractionEventType.LOOT_PICKUP,
        "opponent",
        "last_loot_value",
    ),
    ("host_loot_drops", ExtractionEventType.LOOT_DROP, "host", "last_drop_value"),
    (
        "opponent_loot_drops",
        ExtractionEventType.LOOT_DROP,
        "opponent",
        "last_drop_value",
    ),
    ("host_cache_created", ExtractionEventType.CACHE_CREATED, "host", None),
    (
        "opponent_cache_created",
        ExtractionEventType.CACHE_CREATED,
        "opponent",
        None,
    ),
    ("host_cache_looted", ExtractionEventType.CACHE_LOOTED, "host", "last_loot_value"),
    (
        "opponent_cache_looted",
        ExtractionEventType.CACHE_LOOTED,
        "opponent",
        "last_loot_value",
    ),
    (
        "host_extraction_started",
        ExtractionEventType.EXTRACTION_STARTED,
        "host",
        None,
    ),
    (
        "opponent_extraction_started",
        ExtractionEventType.EXTRACTION_STARTED,
        "opponent",
        None,
    ),
    (
        "host_extraction_interrupted",
        ExtractionEventType.EXTRACTION_INTERRUPTED,
        "host",
        None,
    ),
    (
        "opponent_extraction_interrupted",
        ExtractionEventType.EXTRACTION_INTERRUPTED,
        "opponent",
        None,
    ),
    ("host_extracted", ExtractionEventType.EXTRACTED, "host", None),
    ("opponent_extracted", ExtractionEventType.EXTRACTED, "opponent", None),
    ("host_valid_hits", ExtractionEventType.VALID_HIT, "host", "last_damage"),
    (
        "opponent_valid_hits",
        ExtractionEventType.VALID_HIT,
        "opponent",
        "last_damage",
    ),
    ("host_deaths", ExtractionEventType.DEATH, "host", None),
    ("opponent_deaths", ExtractionEventType.DEATH, "opponent", None),
    ("timeouts", ExtractionEventType.TIMEOUT, "shared", None),
)


class ExtractionEventDecoder:
    def __init__(self) -> None:
        self._previous: ExtractionProtocolSnapshot | None = None

    def reset(self, snapshot: ExtractionProtocolSnapshot | None = None) -> None:
        if snapshot is not None:
            snapshot.validate()
        self._previous = snapshot

    def decode(
        self,
        current: ExtractionProtocolSnapshot,
        *,
        episode_id: int,
        decision_index: int,
    ) -> tuple[ExtractionEvent, ...]:
        current.validate()
        previous = self._previous
        if previous is None:
            self._previous = current
            return ()
        if current.engine_tic < previous.engine_tic:
            raise ValueError("Extraction engine tic decreased")
        events = []
        for name, event_type, side, value_field in _EVENT_COUNTERS:
            delta = int(getattr(current, name)) - int(getattr(previous, name))
            if delta < 0:
                raise ValueError(f"Extraction counter decreased: {name}")
            if delta:
                events.append(
                    ExtractionEvent(
                        type=event_type,
                        side=side,
                        count=delta,
                        value=(
                            0
                            if value_field is None
                            else int(getattr(current, value_field))
                        ),
                        episode_id=episode_id,
                        decision_index=decision_index,
                        engine_tic=current.engine_tic,
                    )
                )
        self._previous = current
        return tuple(events)
