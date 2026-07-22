from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from botcolosseo.envs.duel_protocol import DuelEvent, DuelEventType


@dataclass(frozen=True)
class EventReward:
    weight: float
    cap: int


@dataclass(frozen=True)
class DuelRewardConfig:
    events: dict[DuelEventType, EventReward]


@dataclass(frozen=True)
class DuelRewards:
    host: float
    opponent: float


def load_reward_config(path: Path) -> DuelRewardConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Duel rewards require schema_version 1")
    rewards = {
        DuelEventType(name): EventReward(float(item["weight"]), int(item["cap"]))
        for name, item in payload.get("events", {}).items()
    }
    if set(rewards) != set(DuelEventType):
        missing = set(DuelEventType).difference(rewards)
        raise ValueError(f"Missing duel reward events: {sorted(item.value for item in missing)}")
    if any(item.cap < 0 for item in rewards.values()):
        raise ValueError("Duel reward caps must be nonnegative")
    return DuelRewardConfig(rewards)


class DuelRewardLedger:
    SHAPING_EVENTS = frozenset((DuelEventType.PICKUP, DuelEventType.VALID_HIT))

    def __init__(self, config: DuelRewardConfig) -> None:
        self._config = config
        self.reset()

    def reset(self) -> None:
        self._counts = {
            (side, event_type): 0
            for side in ("host", "opponent")
            for event_type in DuelEventType
        }

    def apply(
        self, events: tuple[DuelEvent, ...], *, shaping_scale: float = 1.0
    ) -> DuelRewards:
        if not 0.0 <= shaping_scale <= 1.0:
            raise ValueError("shaping_scale must be in [0, 1]")
        host_reward = 0.0
        for event in events:
            if event.side not in ("host", "opponent"):
                continue
            rule = self._config.events[event.type]
            key = (event.side, event.type)
            if self._counts[key] >= rule.cap:
                continue
            self._counts[key] += 1
            scale = shaping_scale if event.type in self.SHAPING_EVENTS else 1.0
            weight = rule.weight * scale
            signed = weight if event.side == "host" else -weight
            host_reward += signed
        return DuelRewards(host=host_reward, opponent=-host_reward)
