from __future__ import annotations

from dataclasses import astuple, dataclass, fields
from enum import Enum


class DuelEventType(str, Enum):
    PICKUP = "pickup"
    DROP = "drop"
    SCORE = "score"
    VALID_HIT = "valid_hit"
    DEATH = "death"
    RESPAWN = "respawn"
    CORE_RETURN = "core_return"


@dataclass(frozen=True)
class DuelProtocolSnapshot:
    protocol_version: int
    engine_tic: int
    round_state: int
    carrier: int
    core_state: int
    host_pickups: int
    opponent_pickups: int
    host_drops: int
    opponent_drops: int
    host_scores: int
    opponent_scores: int
    host_valid_hits: int
    opponent_valid_hits: int
    host_deaths: int
    opponent_deaths: int
    host_respawns: int
    opponent_respawns: int
    host_score: int
    opponent_score: int
    winner: int
    core_returns: int
    core_x: int
    core_y: int
    spawn_index: int
    reserved_zero: int

    @classmethod
    def from_values(cls, values: list[int] | tuple[int, ...]) -> DuelProtocolSnapshot:
        if len(values) != len(fields(cls)):
            raise ValueError(f"Duel protocol requires {len(fields(cls))} values")
        snapshot = cls(*(int(value) for value in values))
        snapshot.validate()
        return snapshot

    def to_values(self) -> tuple[int, ...]:
        return astuple(self)

    def validate(self) -> None:
        if self.protocol_version != 2:
            raise ValueError(f"Unsupported duel protocol: {self.protocol_version}")
        if self.engine_tic < 0:
            raise ValueError("engine_tic must be nonnegative")
        if self.round_state not in range(4):
            raise ValueError(f"Invalid round state: {self.round_state}")
        if self.carrier not in range(3):
            raise ValueError(f"Invalid carrier: {self.carrier}")
        if self.core_state not in range(4):
            raise ValueError(f"Invalid core state: {self.core_state}")
        if self.winner not in range(3):
            raise ValueError(f"Invalid winner: {self.winner}")
        if self.spawn_index not in range(3):
            raise ValueError(f"Invalid spawn index: {self.spawn_index}")
        if self.reserved_zero != 0:
            raise ValueError("reserved_zero must remain zero")
        if any(value < 0 for value in self._counter_values()):
            raise ValueError("Duel counters must be nonnegative")

    def _counter_values(self) -> tuple[int, ...]:
        return tuple(getattr(self, name) for name, _, _ in _EVENT_COUNTERS)


@dataclass(frozen=True)
class DuelEvent:
    type: DuelEventType
    side: str
    episode_id: int
    decision_index: int
    engine_tic: int


_EVENT_COUNTERS = (
    ("host_pickups", DuelEventType.PICKUP, "host"),
    ("opponent_pickups", DuelEventType.PICKUP, "opponent"),
    ("host_drops", DuelEventType.DROP, "host"),
    ("opponent_drops", DuelEventType.DROP, "opponent"),
    ("host_scores", DuelEventType.SCORE, "host"),
    ("opponent_scores", DuelEventType.SCORE, "opponent"),
    ("host_valid_hits", DuelEventType.VALID_HIT, "host"),
    ("opponent_valid_hits", DuelEventType.VALID_HIT, "opponent"),
    ("host_deaths", DuelEventType.DEATH, "host"),
    ("opponent_deaths", DuelEventType.DEATH, "opponent"),
    ("host_respawns", DuelEventType.RESPAWN, "host"),
    ("opponent_respawns", DuelEventType.RESPAWN, "opponent"),
    ("core_returns", DuelEventType.CORE_RETURN, "shared"),
)


class DuelEventDecoder:
    def __init__(self) -> None:
        self._previous: DuelProtocolSnapshot | None = None

    def reset(self, snapshot: DuelProtocolSnapshot | None = None) -> None:
        if snapshot is not None:
            snapshot.validate()
        self._previous = snapshot

    def decode(
        self,
        current: DuelProtocolSnapshot,
        *,
        episode_id: int,
        decision_index: int,
    ) -> tuple[DuelEvent, ...]:
        current.validate()
        previous = self._previous
        if previous is None:
            self._previous = current
            return ()
        if current.engine_tic < previous.engine_tic:
            raise ValueError("Duel engine tic decreased")
        events: list[DuelEvent] = []
        for name, event_type, side in _EVENT_COUNTERS:
            delta = getattr(current, name) - getattr(previous, name)
            if delta < 0:
                raise ValueError(f"Duel counter decreased: {name}")
            if delta > 1:
                raise ValueError(f"Duel counter jumped: {name}")
            if delta:
                events.append(
                    DuelEvent(
                        type=event_type,
                        side=side,
                        episode_id=episode_id,
                        decision_index=decision_index,
                        engine_tic=current.engine_tic,
                    )
                )
        self._previous = current
        return tuple(events)
