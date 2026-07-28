from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_protocol import ExtractionEvent, ExtractionEventType
from botcolosseo.envs.extraction_types import (
    ExtractionActorObservation,
    ExtractionPrivilegedState,
)

ATTACK_ACTIONS = frozenset(
    {
        MacroAction.ATTACK,
        MacroAction.FORWARD_ATTACK,
        MacroAction.TURN_LEFT_ATTACK,
        MacroAction.TURN_RIGHT_ATTACK,
    }
)


@dataclass(frozen=True)
class ExtractionReward:
    total: float
    components: dict[str, float]


class _BoundedLedger:
    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    def _add(
        self,
        components: dict[str, float],
        name: str,
        value: float,
        cap: int,
    ) -> None:
        if self._counts[name] >= cap:
            return
        self._counts[name] += 1
        components[name] = components.get(name, 0.0) + value

    @staticmethod
    def _result(components: dict[str, float], scale: float) -> ExtractionReward:
        scaled = {name: value * scale for name, value in components.items()}
        return ExtractionReward(sum(scaled.values()), scaled)


@dataclass(frozen=True)
class ExtractionTaskRewardConfig:
    loot_value_scale: float = 0.002
    extraction_started: float = 0.02
    invalid_attack: float = -0.002
    death: float = -0.25
    death_value_scale: float = -1 / 150
    timeout_value_scale: float = -1 / 150
    loot_cap: int = 8
    extraction_started_cap: int = 2
    invalid_attack_cap: int = 30


class ExtractionTaskRewardLedger(_BoundedLedger):
    """Decaying progress shaping; terminal outcome remains environment-owned."""

    def __init__(self, config: ExtractionTaskRewardConfig, *, learner_side: str) -> None:
        if learner_side not in {"host", "opponent"}:
            raise ValueError("Extraction learner side is invalid")
        super().__init__()
        self.config = config
        self.learner_side = learner_side

    def apply(
        self,
        action: MacroAction,
        events: tuple[ExtractionEvent, ...],
        *,
        observation_before: ExtractionActorObservation,
        state_before: ExtractionPrivilegedState,
        state_after: ExtractionPrivilegedState,
        scale: float,
    ) -> ExtractionReward:
        del state_before
        if not 0 <= scale <= 1:
            raise ValueError("Extraction task shaping scale must be in [0, 1]")
        components: dict[str, float] = {}
        learner_events = tuple(event for event in events if event.side == self.learner_side)
        valid_hit = any(
            event.type is ExtractionEventType.VALID_HIT for event in learner_events
        )
        for event in learner_events:
            if event.type in {
                ExtractionEventType.LOOT_PICKUP,
                ExtractionEventType.CACHE_LOOTED,
            }:
                self._add(
                    components,
                    "loot_progress",
                    event.value * self.config.loot_value_scale,
                    self.config.loot_cap,
                )
            elif event.type is ExtractionEventType.EXTRACTION_STARTED:
                self._add(
                    components,
                    "extraction_started",
                    self.config.extraction_started,
                    self.config.extraction_started_cap,
                )
            elif event.type is ExtractionEventType.DEATH:
                components["death"] = self.config.death
                components["death_value_loss"] = (
                    observation_before.carried_value
                    * self.config.death_value_scale
                )
        if MacroAction(action) in ATTACK_ACTIONS and not valid_hit:
            self._add(
                components,
                "invalid_attack",
                self.config.invalid_attack,
                self.config.invalid_attack_cap,
            )
        if any(event.type is ExtractionEventType.TIMEOUT for event in events):
            slots = (
                state_after.host_slots
                if self.learner_side == "host"
                else state_after.opponent_slots
            )
            components["timeout_value_loss"] = (
                sum(slots) * self.config.timeout_value_scale
            )
        dense_components = {"loot_progress", "extraction_started"}
        scaled = {
            name: value * scale if name in dense_components else value
            for name, value in components.items()
        }
        return ExtractionReward(sum(scaled.values()), scaled)


@dataclass(frozen=True)
class AggressiveExtractionRewardConfig:
    valid_hit: float = 0.03
    engagement_initiation: float = 0.05
    cache_looted: float = 0.10
    cache_to_extraction: float = 0.25
    invalid_attack: float = -0.005
    low_resource_attack: float = -0.01
    valid_hit_cap: int = 10
    initiation_cap: int = 4
    cache_looted_cap: int = 3
    invalid_attack_cap: int = 30
    low_resource_attack_cap: int = 12
    initiation_cooldown: int = 12


class AggressiveExtractionRewardLedger(_BoundedLedger):
    def __init__(
        self,
        config: AggressiveExtractionRewardConfig,
        *,
        learner_side: str,
        scale: float,
    ) -> None:
        if learner_side not in {"host", "opponent"} or scale < 0:
            raise ValueError("Aggressive Extraction reward inputs are invalid")
        super().__init__()
        self.config = config
        self.learner_side = learner_side
        self.scale = scale
        self._since_hit = config.initiation_cooldown + 1
        self._killed_opponent = False
        self._looted_cache = False

    def apply(
        self,
        action: MacroAction,
        events: tuple[ExtractionEvent, ...],
        *,
        observation_before: ExtractionActorObservation,
        state_before: ExtractionPrivilegedState,
        state_after: ExtractionPrivilegedState,
    ) -> ExtractionReward:
        del state_before, state_after
        components: dict[str, float] = {}
        learner_events = tuple(event for event in events if event.side == self.learner_side)
        if any(
            event.type is ExtractionEventType.DEATH
            and event.side != self.learner_side
            for event in events
        ):
            self._killed_opponent = True
        valid_hit = any(
            event.type is ExtractionEventType.VALID_HIT for event in learner_events
        )
        if valid_hit and observation_before.health >= 40 and observation_before.ammo > 0:
            self._add(
                components,
                "valid_hit",
                self.config.valid_hit,
                self.config.valid_hit_cap,
            )
            if self._since_hit > self.config.initiation_cooldown:
                self._add(
                    components,
                    "engagement_initiation",
                    self.config.engagement_initiation,
                    self.config.initiation_cap,
                )
            self._since_hit = 0
        else:
            self._since_hit += 1
        for event in learner_events:
            if (
                event.type is ExtractionEventType.CACHE_LOOTED
                and self._killed_opponent
            ):
                self._looted_cache = True
                self._add(
                    components,
                    "cache_looted",
                    self.config.cache_looted,
                    self.config.cache_looted_cap,
                )
            elif (
                event.type is ExtractionEventType.EXTRACTED and self._looted_cache
            ):
                components["cache_to_extraction"] = self.config.cache_to_extraction
        if MacroAction(action) in ATTACK_ACTIONS and not valid_hit:
            self._add(
                components,
                "invalid_attack",
                self.config.invalid_attack,
                self.config.invalid_attack_cap,
            )
            if observation_before.health < 40 or observation_before.ammo <= 5:
                self._add(
                    components,
                    "low_resource_attack",
                    self.config.low_resource_attack,
                    self.config.low_resource_attack_cap,
                )
        return self._result(components, self.scale)


@dataclass(frozen=True)
class DefensiveExtractionRewardConfig:
    risk_disengagement: float = 0.10
    meaningful_extraction: float = 0.20
    combat_with_value: float = -0.015
    empty_idle: float = -0.003
    risk_disengagement_cap: int = 3
    combat_with_value_cap: int = 12
    empty_idle_cap: int = 20


class DefensiveExtractionRewardLedger(_BoundedLedger):
    def __init__(
        self,
        config: DefensiveExtractionRewardConfig,
        *,
        learner_side: str,
        scale: float,
    ) -> None:
        if learner_side not in {"host", "opponent"} or scale < 0:
            raise ValueError("Defensive Extraction reward inputs are invalid")
        super().__init__()
        self.config = config
        self.learner_side = learner_side
        self.scale = scale
        self._risk_active = False

    def apply(
        self,
        action: MacroAction,
        events: tuple[ExtractionEvent, ...],
        *,
        observation_before: ExtractionActorObservation,
        state_before: ExtractionPrivilegedState,
        state_after: ExtractionPrivilegedState,
    ) -> ExtractionReward:
        components: dict[str, float] = {}
        before_distance = _opponent_distance(state_before, self.learner_side)
        after_distance = _opponent_distance(state_after, self.learner_side)
        low_resources = observation_before.health <= 40 or observation_before.ammo <= 5
        self._risk_active |= low_resources and before_distance <= 384
        if self._risk_active and after_distance >= 512:
            self._add(
                components,
                "risk_disengagement",
                self.config.risk_disengagement,
                self.config.risk_disengagement_cap,
            )
            self._risk_active = False
        learner_events = tuple(event for event in events if event.side == self.learner_side)
        if (
            observation_before.carried_value >= 25
            and any(event.type is ExtractionEventType.EXTRACTED for event in learner_events)
        ):
            components["meaningful_extraction"] = self.config.meaningful_extraction
        if (
            observation_before.carried_value >= 50
            and MacroAction(action) in ATTACK_ACTIONS
        ):
            self._add(
                components,
                "combat_with_value",
                self.config.combat_with_value,
                self.config.combat_with_value_cap,
            )
        if observation_before.carried_value == 0 and MacroAction(action) is MacroAction.IDLE:
            self._add(
                components,
                "empty_idle",
                self.config.empty_idle,
                self.config.empty_idle_cap,
            )
        return self._result(components, self.scale)


@dataclass(frozen=True)
class ExplorerExtractionRewardConfig:
    novel_loot_region: float = 0.04
    backpack_upgrade: float = 0.10
    upgrade_to_extraction: float = 0.20
    novel_loot_region_cap: int = 7
    backpack_upgrade_cap: int = 3


class ExplorerExtractionRewardLedger(_BoundedLedger):
    def __init__(
        self,
        config: ExplorerExtractionRewardConfig,
        *,
        learner_side: str,
        scale: float,
    ) -> None:
        if learner_side not in {"host", "opponent"} or scale < 0:
            raise ValueError("Explorer Extraction reward inputs are invalid")
        super().__init__()
        self.config = config
        self.learner_side = learner_side
        self.scale = scale
        self._loot_cells: set[tuple[int, int]] = set()
        self._upgraded = False

    def apply(
        self,
        action: MacroAction,
        events: tuple[ExtractionEvent, ...],
        *,
        observation_before: ExtractionActorObservation,
        state_before: ExtractionPrivilegedState,
        state_after: ExtractionPrivilegedState,
    ) -> ExtractionReward:
        del action, observation_before, state_before
        components: dict[str, float] = {}
        learner_events = tuple(event for event in events if event.side == self.learner_side)
        position = _learner_position(state_after, self.learner_side)
        cell = (math.floor(position[0] / 160), math.floor(position[1] / 160))
        if (
            any(event.type is ExtractionEventType.LOOT_PICKUP for event in learner_events)
            and cell not in self._loot_cells
        ):
            self._loot_cells.add(cell)
            self._add(
                components,
                "novel_loot_region",
                self.config.novel_loot_region,
                self.config.novel_loot_region_cap,
            )
        if any(event.type is ExtractionEventType.LOOT_DROP for event in learner_events):
            self._upgraded = True
            self._add(
                components,
                "backpack_upgrade",
                self.config.backpack_upgrade,
                self.config.backpack_upgrade_cap,
            )
        if self._upgraded and any(
            event.type is ExtractionEventType.EXTRACTED for event in learner_events
        ):
            components["upgrade_to_extraction"] = self.config.upgrade_to_extraction
        return self._result(components, self.scale)


def _learner_position(
    state: ExtractionPrivilegedState, side: str
) -> tuple[float, float]:
    if side == "host":
        return state.host_x, state.host_y
    return state.opponent_x, state.opponent_y


def _opponent_distance(state: ExtractionPrivilegedState, side: str) -> float:
    return math.dist(
        _learner_position(state, side),
        _learner_position(state, "opponent" if side == "host" else "host"),
    )
