from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from botcolosseo.agents.extraction_teachers import (
    opponent_health,
    opponent_position,
    player_health,
    player_pose,
    player_slots,
    steer_toward,
)
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_layouts import randomized_loot_layout
from botcolosseo.envs.extraction_protocol import ExtractionEvent, ExtractionEventType
from botcolosseo.envs.extraction_types import (
    ExtractionActorObservation,
    ExtractionPrivilegedState,
)
from botcolosseo.training.extraction_rewards import (
    ATTACK_ACTIONS,
    MEANINGFUL_VALUE,
    ExtractionReward,
)


@dataclass(frozen=True)
class StyleOpportunity:
    active: bool
    preferred_actions: frozenset[MacroAction]
    phase: str

    def __post_init__(self) -> None:
        if self.active != bool(self.preferred_actions):
            raise ValueError("Opportunity activity and preferred actions disagree")


@dataclass(frozen=True)
class OpportunityShapingConfig:
    gamma: float = 0.99
    pbrs_scale: float = 0.10
    completion_utility: float = 0.30
    invalid_action_penalty: float = -0.005
    invalid_action_cap: int = 24

    def __post_init__(self) -> None:
        if (
            not 0 <= self.gamma <= 1
            or self.pbrs_scale < 0
            or self.completion_utility < 0
            or self.invalid_action_penalty > 0
            or self.invalid_action_cap < 0
        ):
            raise ValueError("Opportunity shaping configuration is invalid")


@dataclass(frozen=True)
class AggressiveOpportunityConfig(OpportunityShapingConfig):
    engagement_distance: float = 768.0
    attack_distance: float = 512.0
    minimum_health: float = 40.0
    minimum_ammo: float = 6.0
    maximum_carried_value: int = 25

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.engagement_distance <= 0
            or not 0 < self.attack_distance <= self.engagement_distance
            or self.minimum_health < 0
            or self.minimum_ammo < 0
            or self.maximum_carried_value < 0
        ):
            raise ValueError("Aggressive opportunity configuration is invalid")


@dataclass(frozen=True)
class DefensiveOpportunityConfig(OpportunityShapingConfig):
    risk_distance: float = 512.0
    disengaged_distance: float = 640.0
    risk_health: float = 50.0
    risk_ammo: float = 5.0
    meaningful_value: int = MEANINGFUL_VALUE

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.risk_distance <= 0
            or self.disengaged_distance <= self.risk_distance
            or self.risk_health < 0
            or self.risk_ammo < 0
            or self.meaningful_value <= 0
        ):
            raise ValueError("Defensive opportunity configuration is invalid")


@dataclass(frozen=True)
class ExplorerOpportunityConfig(OpportunityShapingConfig):
    extraction_value: int = 75
    urgent_time: float = 20.0
    required_novel_regions: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.extraction_value <= 0
            or self.urgent_time < 0
            or self.required_novel_regions <= 0
        ):
            raise ValueError("Explorer opportunity configuration is invalid")


class _OpportunityLedger:
    _POTENTIALS = (0.0, 0.20, 0.40, 0.60, 0.80)

    def __init__(
        self,
        config: OpportunityShapingConfig,
        *,
        learner_side: str,
        scale: float,
    ) -> None:
        if learner_side not in {"host", "opponent"} or scale < 0:
            raise ValueError("Opportunity ledger inputs are invalid")
        self.config = config
        self.learner_side = learner_side
        self.scale = scale
        self._phase = 0
        self._counts: Counter[str] = Counter()

    def _bounded(
        self, components: dict[str, float], name: str, value: float, cap: int
    ) -> None:
        if self._counts[name] >= cap:
            return
        self._counts[name] += 1
        components[name] = components.get(name, 0.0) + value

    def _finish(
        self,
        components: dict[str, float],
        *,
        phase_after: int,
        terminal: bool,
        completed: bool,
    ) -> ExtractionReward:
        if not 0 <= phase_after < len(self._POTENTIALS):
            raise ValueError("Opportunity phase is outside the potential table")
        potential_before = self._POTENTIALS[self._phase]
        potential_after = 0.0 if terminal else self._POTENTIALS[phase_after]
        shaping = self.config.pbrs_scale * (
            self.config.gamma * potential_after - potential_before
        )
        if abs(shaping) > 1e-12:
            components["pbrs_progress"] = shaping
        if completed:
            components["completion_utility"] = self.config.completion_utility
        self._phase = 0 if terminal else phase_after
        scaled = {name: value * self.scale for name, value in components.items()}
        return ExtractionReward(sum(scaled.values()), scaled)

    def _terminal(self, events: tuple[ExtractionEvent, ...]) -> bool:
        return any(
            event.type in {ExtractionEventType.EXTRACTED, ExtractionEventType.TIMEOUT}
            or (
                event.type is ExtractionEventType.DEATH
                and event.side == self.learner_side
            )
            for event in events
        )


class AggressiveOpportunityLedger(_OpportunityLedger):
    def __init__(
        self,
        config: AggressiveOpportunityConfig,
        *,
        learner_side: str,
        scale: float,
    ) -> None:
        super().__init__(config, learner_side=learner_side, scale=scale)
        self.config = config
        self._killed_opponent = False
        self._looted_cache = False

    def opportunity(
        self,
        *,
        observation_before: ExtractionActorObservation,
        state_before: ExtractionPrivilegedState,
    ) -> StyleOpportunity:
        own_x, own_y, _ = player_pose(state_before, self.learner_side)
        target = opponent_position(state_before, self.learner_side)
        distance = math.hypot(target[0] - own_x, target[1] - own_y)
        active = (
            player_health(state_before, self.learner_side) > 0
            and opponent_health(state_before, self.learner_side) > 0
            and distance <= self.config.engagement_distance
            and observation_before.health >= self.config.minimum_health
            and observation_before.ammo >= self.config.minimum_ammo
            and observation_before.carried_value <= self.config.maximum_carried_value
        )
        if not active:
            return StyleOpportunity(False, frozenset(), "no_favorable_engagement")
        preferred = steer_toward(
            state_before,
            self.learner_side,
            target,
            attack=distance <= self.config.attack_distance,
        )
        return StyleOpportunity(True, frozenset({preferred}), "favorable_engagement")

    def apply(
        self,
        action: MacroAction,
        events: tuple[ExtractionEvent, ...],
        *,
        observation_before: ExtractionActorObservation,
        state_before: ExtractionPrivilegedState,
        state_after: ExtractionPrivilegedState,
    ) -> ExtractionReward:
        del state_after
        components: dict[str, float] = {}
        phase_after = self._phase
        opportunity = self.opportunity(
            observation_before=observation_before, state_before=state_before
        )
        learner_events = tuple(event for event in events if event.side == self.learner_side)
        valid_hit = any(
            event.type is ExtractionEventType.VALID_HIT for event in learner_events
        )
        if opportunity.active and MacroAction(action) in opportunity.preferred_actions:
            phase_after = max(phase_after, 1)
        if valid_hit and opportunity.active:
            phase_after = max(phase_after, 2)
        if any(
            event.type is ExtractionEventType.DEATH
            and event.side != self.learner_side
            for event in events
        ):
            self._killed_opponent = True
            phase_after = max(phase_after, 3)
        if self._killed_opponent and any(
            event.type is ExtractionEventType.CACHE_LOOTED for event in learner_events
        ):
            self._looted_cache = True
            phase_after = 4
        completed = self._looted_cache and any(
            event.type is ExtractionEventType.EXTRACTED for event in learner_events
        )
        if MacroAction(action) in ATTACK_ACTIONS and not valid_hit:
            self._bounded(
                components,
                "invalid_attack",
                self.config.invalid_action_penalty,
                self.config.invalid_action_cap,
            )
        terminal = self._terminal(events)
        return self._finish(
            components,
            phase_after=phase_after,
            terminal=terminal,
            completed=completed,
        )


class DefensiveOpportunityLedger(_OpportunityLedger):
    def __init__(
        self,
        config: DefensiveOpportunityConfig,
        *,
        learner_side: str,
        scale: float,
    ) -> None:
        super().__init__(config, learner_side=learner_side, scale=scale)
        self.config = config

    def opportunity(
        self,
        *,
        observation_before: ExtractionActorObservation,
        state_before: ExtractionPrivilegedState,
    ) -> StyleOpportunity:
        own_x, own_y, _ = player_pose(state_before, self.learner_side)
        enemy = opponent_position(state_before, self.learner_side)
        distance = math.hypot(enemy[0] - own_x, enemy[1] - own_y)
        resource_risk = (
            observation_before.health <= self.config.risk_health
            or observation_before.ammo <= self.config.risk_ammo
            or observation_before.carried_value >= self.config.meaningful_value
        )
        active = (
            player_health(state_before, self.learner_side) > 0
            and opponent_health(state_before, self.learner_side) > 0
            and distance <= self.config.risk_distance
            and resource_risk
        )
        if not active:
            return StyleOpportunity(False, frozenset(), "no_material_risk")
        extraction = (0.0, 400.0 if self.learner_side == "host" else -400.0)
        preferred = {steer_toward(state_before, self.learner_side, extraction)}
        if distance <= 256.0:
            preferred.add(MacroAction.MOVE_BACKWARD)
        return StyleOpportunity(True, frozenset(preferred), "material_risk")

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
        phase_after = self._phase
        opportunity = self.opportunity(
            observation_before=observation_before, state_before=state_before
        )
        if opportunity.active and MacroAction(action) in opportunity.preferred_actions:
            phase_after = max(phase_after, 1)
        own_after = player_pose(state_after, self.learner_side)
        enemy_after = opponent_position(state_after, self.learner_side)
        after_distance = math.hypot(
            enemy_after[0] - own_after[0], enemy_after[1] - own_after[1]
        )
        if phase_after >= 1 and after_distance >= self.config.disengaged_distance:
            phase_after = max(phase_after, 2)
        learner_events = tuple(event for event in events if event.side == self.learner_side)
        if phase_after >= 1 and any(
            event.type is ExtractionEventType.EXTRACTION_STARTED for event in learner_events
        ):
            phase_after = 3
        completed = (
            phase_after >= 1
            and observation_before.carried_value >= self.config.meaningful_value
            and any(event.type is ExtractionEventType.EXTRACTED for event in learner_events)
        )
        if opportunity.active and MacroAction(action) in ATTACK_ACTIONS:
            self._bounded(
                components,
                "risk_escalation",
                self.config.invalid_action_penalty,
                self.config.invalid_action_cap,
            )
        if observation_before.carried_value == 0 and MacroAction(action) is MacroAction.IDLE:
            self._bounded(
                components,
                "empty_idle",
                self.config.invalid_action_penalty,
                self.config.invalid_action_cap,
            )
        return self._finish(
            components,
            phase_after=phase_after,
            terminal=self._terminal(events),
            completed=completed,
        )


class ExplorerOpportunityLedger(_OpportunityLedger):
    def __init__(
        self,
        config: ExplorerOpportunityConfig,
        *,
        learner_side: str,
        scale: float,
        layout_variant: int,
    ) -> None:
        super().__init__(config, learner_side=learner_side, scale=scale)
        self.config = config
        self._layout = randomized_loot_layout(layout_variant)
        self._target_id: int | None = None
        self._visited_cells: set[tuple[int, int]] = set()
        self._upgraded = False

    def _useful_loot_ids(
        self, state: ExtractionPrivilegedState
    ) -> tuple[int, ...]:
        minimum = min(player_slots(state, self.learner_side))
        return tuple(
            loot_id
            for loot_id, (value, _, _) in enumerate(self._layout)
            if state.world_loot_mask & (1 << loot_id) and value > minimum
        )

    def _target(self, state: ExtractionPrivilegedState) -> int | None:
        useful = self._useful_loot_ids(state)
        if self._target_id not in useful:
            self._target_id = None
        if self._target_id is None and useful:
            own_x, own_y, _ = player_pose(state, self.learner_side)
            minimum = min(player_slots(state, self.learner_side))

            def priority(loot_id: int) -> tuple[float, ...]:
                value, x, y = self._layout[loot_id]
                cell = (math.floor(x / 160), math.floor(y / 160))
                return (
                    0 if cell not in self._visited_cells else 1,
                    -(value - minimum),
                    -math.hypot(x - own_x, y - own_y),
                    loot_id,
                )

            self._target_id = min(useful, key=priority)
        return self._target_id

    def opportunity(
        self,
        *,
        observation_before: ExtractionActorObservation,
        state_before: ExtractionPrivilegedState,
    ) -> StyleOpportunity:
        if player_health(state_before, self.learner_side) <= 0:
            return StyleOpportunity(False, frozenset(), "inactive")
        extraction = (0.0, 400.0 if self.learner_side == "host" else -400.0)
        should_convert = (
            self._upgraded
            or observation_before.carried_value >= self.config.extraction_value
            or observation_before.remaining_time <= self.config.urgent_time
        )
        if should_convert:
            action = steer_toward(state_before, self.learner_side, extraction)
            return StyleOpportunity(True, frozenset({action}), "convert_exploration")
        target_id = self._target(state_before)
        if target_id is None:
            return StyleOpportunity(False, frozenset(), "no_useful_novel_loot")
        _, target_x, target_y = self._layout[target_id]
        action = steer_toward(
            state_before, self.learner_side, (target_x, target_y)
        )
        return StyleOpportunity(True, frozenset({action}), "novel_loot_route")

    def apply(
        self,
        action: MacroAction,
        events: tuple[ExtractionEvent, ...],
        *,
        observation_before: ExtractionActorObservation,
        state_before: ExtractionPrivilegedState,
        state_after: ExtractionPrivilegedState,
    ) -> ExtractionReward:
        del state_before
        components: dict[str, float] = {}
        phase_after = self._phase
        learner_events = tuple(event for event in events if event.side == self.learner_side)
        if any(event.type is ExtractionEventType.LOOT_PICKUP for event in learner_events):
            x, y, _ = player_pose(state_after, self.learner_side)
            cell = (math.floor(x / 160), math.floor(y / 160))
            if cell not in self._visited_cells:
                self._visited_cells.add(cell)
                phase_after = max(phase_after, 1)
        if any(event.type is ExtractionEventType.LOOT_DROP for event in learner_events):
            self._upgraded = True
            phase_after = max(phase_after, 2)
        if phase_after >= 1 and any(
            event.type is ExtractionEventType.EXTRACTION_STARTED for event in learner_events
        ):
            phase_after = 3
        completed = (
            len(self._visited_cells) >= self.config.required_novel_regions
            and any(event.type is ExtractionEventType.EXTRACTED for event in learner_events)
        )
        if observation_before.carried_value == 0 and MacroAction(action) is MacroAction.IDLE:
            self._bounded(
                components,
                "empty_idle",
                self.config.invalid_action_penalty,
                self.config.invalid_action_cap,
            )
        return self._finish(
            components,
            phase_after=phase_after,
            terminal=self._terminal(events),
            completed=completed,
        )
