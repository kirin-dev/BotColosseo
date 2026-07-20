from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import ClassVar

import vizdoom as vzd

PROTOCOL_VERSION = 1
COUNTER_FIELDS = (
    "pickup_count",
    "drop_count",
    "score_count",
    "valid_hit_count",
    "death_count",
    "respawn_count",
    "core_return_count",
    "task_success_count",
)


class EventProtocolError(RuntimeError):
    """Raised when ACS state violates the versioned event protocol."""


class EventType(str, Enum):
    PICKUP = "pickup"
    DROP = "drop"
    SCORE = "score"
    VALID_HIT = "valid_hit"
    DEATH = "death"
    RESPAWN = "respawn"
    CORE_RETURN = "core_return"
    TASK_SUCCESS = "task_success"
    REGION_TRANSITION = "region_transition"


@dataclass(frozen=True)
class EpisodeEvent:
    episode_id: int
    engine_tic: int
    decision_index: int
    type: EventType
    subject: str = "agent"
    region_from: str | None = None
    region_to: str | None = None
    value: float = 1.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass(frozen=True)
class ProtocolSnapshot:
    protocol_version: int
    engine_tic: int
    task_phase: int
    core_state: int
    pickup_count: int
    drop_count: int
    score_count: int
    valid_hit_count: int
    death_count: int
    respawn_count: int
    core_return_count: int
    task_success_count: int
    target_state: int
    home_score: int
    away_score: int
    reserved_zero: int
    core_x: float
    core_y: float
    target_x: float
    target_y: float

    _FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "protocol_version",
        "engine_tic",
        "task_phase",
        "core_state",
        *COUNTER_FIELDS,
        "target_state",
        "home_score",
        "away_score",
        "reserved_zero",
        "core_x",
        "core_y",
        "target_x",
        "target_y",
    )

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return cls._FIELD_NAMES

    @classmethod
    def from_values(cls, values) -> ProtocolSnapshot:
        values = tuple(values)
        if len(values) != 20:
            raise EventProtocolError(f"Expected 20 USER variables, got {len(values)}")
        integers = [int(value) for value in values[:16]]
        coordinates = [vzd.doom_fixed_to_float(int(value)) for value in values[16:]]
        return cls(*integers, *coordinates)


_COUNTER_EVENTS = {
    "pickup_count": EventType.PICKUP,
    "drop_count": EventType.DROP,
    "score_count": EventType.SCORE,
    "valid_hit_count": EventType.VALID_HIT,
    "death_count": EventType.DEATH,
    "respawn_count": EventType.RESPAWN,
    "core_return_count": EventType.CORE_RETURN,
    "task_success_count": EventType.TASK_SUCCESS,
}


class EventDecoder:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._previous: ProtocolSnapshot | None = None
        self._previous_region: str | None = None

    def decode(
        self,
        snapshot: ProtocolSnapshot,
        *,
        region_name: str | None,
        episode_id: int,
        decision_index: int,
    ) -> tuple[EpisodeEvent, ...]:
        self._validate_static(snapshot)
        previous = self._previous
        if previous is None:
            self._previous = snapshot
            self._previous_region = region_name
            return ()
        if snapshot.engine_tic < previous.engine_tic:
            raise EventProtocolError("engine tic decreased inside an episode")

        events: list[EpisodeEvent] = []
        for field_name in COUNTER_FIELDS:
            delta = getattr(snapshot, field_name) - getattr(previous, field_name)
            if delta < 0:
                raise EventProtocolError(f"counter decreased: {field_name}")
            if delta > 1:
                raise EventProtocolError(f"counter jumped by more than one: {field_name}")
            if delta == 1:
                events.append(
                    EpisodeEvent(
                        episode_id=episode_id,
                        engine_tic=snapshot.engine_tic,
                        decision_index=decision_index,
                        type=_COUNTER_EVENTS[field_name],
                    )
                )

        if (
            region_name is not None
            and self._previous_region is not None
            and region_name != self._previous_region
        ):
            events.append(
                EpisodeEvent(
                    episode_id=episode_id,
                    engine_tic=snapshot.engine_tic,
                    decision_index=decision_index,
                    type=EventType.REGION_TRANSITION,
                    region_from=self._previous_region,
                    region_to=region_name,
                )
            )

        self._previous = snapshot
        if region_name is not None:
            self._previous_region = region_name
        return tuple(events)

    @staticmethod
    def _validate_static(snapshot: ProtocolSnapshot) -> None:
        if snapshot.protocol_version != PROTOCOL_VERSION:
            raise EventProtocolError(
                f"protocol version {snapshot.protocol_version}, expected {PROTOCOL_VERSION}"
            )
        if snapshot.reserved_zero != 0:
            raise EventProtocolError("reserved USER16 must remain zero")
