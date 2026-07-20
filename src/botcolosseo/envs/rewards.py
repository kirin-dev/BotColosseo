from __future__ import annotations

from dataclasses import dataclass

from botcolosseo.envs.events import EpisodeEvent, EventType
from botcolosseo.scenarios.regions import RegionGraph


@dataclass(frozen=True)
class RewardConfig:
    pickup: float = 0.25
    score: float = 1.0
    valid_hit: float = 0.05
    progress: float = 0.01
    pickup_cap: int = 1
    score_cap: int = 3
    valid_hit_cap: int = 5
    progress_cap: int = 25


class RewardLedger:
    def __init__(self, graph: RegionGraph, config: RewardConfig | None = None) -> None:
        self._graph = graph
        self._config = config or RewardConfig()
        self.reset()

    def reset(self) -> None:
        self._counts = {
            EventType.PICKUP: 0,
            EventType.SCORE: 0,
            EventType.VALID_HIT: 0,
            EventType.REGION_TRANSITION: 0,
        }

    def apply(self, events: tuple[EpisodeEvent, ...], *, target_region: str) -> float:
        reward = 0.0
        for event in events:
            if event.type is EventType.PICKUP:
                reward += self._bounded(
                    EventType.PICKUP,
                    self._config.pickup,
                    self._config.pickup_cap,
                )
            elif event.type is EventType.SCORE:
                reward += self._bounded(EventType.SCORE, self._config.score, self._config.score_cap)
            elif event.type is EventType.VALID_HIT:
                reward += self._bounded(
                    EventType.VALID_HIT,
                    self._config.valid_hit,
                    self._config.valid_hit_cap,
                )
            elif event.type is EventType.REGION_TRANSITION and self._is_progress(
                event, target_region
            ):
                reward += self._bounded(
                    EventType.REGION_TRANSITION,
                    self._config.progress,
                    self._config.progress_cap,
                )
        return reward

    def _bounded(self, event_type: EventType, value: float, cap: int) -> float:
        count = self._counts[event_type]
        if count >= cap:
            return 0.0
        self._counts[event_type] = count + 1
        return value

    def _is_progress(self, event: EpisodeEvent, target_region: str) -> bool:
        if event.region_from is None or event.region_to is None:
            return False
        before = len(self._graph.shortest_path(event.region_from, target_region)) - 1
        after = len(self._graph.shortest_path(event.region_to, target_region)) - 1
        return after < before
