from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from botcolosseo.envs.actions import MacroAction

ACTION_COUNT = len(MacroAction)
ZERO_BIAS = (0.0,) * ACTION_COUNT

DefensiveState = Literal["base", "guard", "disengage", "recover"]
ExplorerState = Literal["base", "route_commit", "stall_recovery"]
ExplorerMode = Literal["upper", "lower", "flank"]


@dataclass(frozen=True)
class PublicStyleContext:
    health: float
    armor: float
    ammo: float
    own_score: int
    opponent_score: int
    has_core: bool
    previous_action: MacroAction
    base_logits: tuple[float, ...]
    decision_index: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.health <= 200.0:
            raise ValueError("Public health is out of range")
        if not 0.0 <= self.armor <= 200.0 or self.ammo < 0.0:
            raise ValueError("Public armor or ammo is invalid")
        if self.own_score < 0 or self.opponent_score < 0 or self.decision_index < 0:
            raise ValueError("Public counters must be nonnegative")
        if len(self.base_logits) != ACTION_COUNT or not all(
            math.isfinite(value) for value in self.base_logits
        ):
            raise ValueError("Base logits must contain one finite value per action")
        MacroAction(self.previous_action)


@dataclass(frozen=True)
class GovernorDecision:
    state: str
    trigger: str
    reason: str
    logit_bias: tuple[float, ...] = ZERO_BIAS
    override_action: MacroAction | None = None
    max_remaining_interventions: int = 0
    fallback_condition: str = "inactive"
    route_mode: ExplorerMode | None = None

    def __post_init__(self) -> None:
        if not self.state or not self.trigger or not self.reason or not self.fallback_condition:
            raise ValueError("Governor telemetry strings must be non-empty")
        if len(self.logit_bias) != ACTION_COUNT or not all(
            math.isfinite(value) for value in self.logit_bias
        ):
            raise ValueError("Governor bias must contain one finite value per action")
        if self.max_remaining_interventions < 0:
            raise ValueError("Remaining interventions must be nonnegative")
        if self.override_action is not None:
            MacroAction(self.override_action)

    @property
    def intervened(self) -> bool:
        return self.override_action is not None or self.logit_bias != ZERO_BIAS


@dataclass(frozen=True)
class GovernorTelemetry:
    decision_index: int
    base_action: MacroAction
    final_action: MacroAction
    state: str
    trigger: str
    reason: str
    intervened: bool
    used_override: bool
    fallback_condition: str
    route_mode: ExplorerMode | None


@dataclass(frozen=True)
class DefensiveGovernorConfig:
    guard_decisions: int
    guard_bias: float
    disengage_decisions: int
    disengage_bias: float
    recover_decisions: int
    low_health_threshold: float
    health_drop_threshold: float
    max_consecutive_interventions: int

    def __post_init__(self) -> None:
        positive_ints = (
            self.guard_decisions,
            self.disengage_decisions,
            self.recover_decisions,
            self.max_consecutive_interventions,
        )
        if any(type(value) is not int or value <= 0 for value in positive_ints):
            raise ValueError("Defensive decision limits must be positive integers")
        if self.guard_decisions > self.max_consecutive_interventions:
            raise ValueError("Guard exceeds the consecutive intervention limit")
        if self.disengage_decisions > self.max_consecutive_interventions:
            raise ValueError("Disengage exceeds the consecutive intervention limit")
        if not 0.0 < self.low_health_threshold <= 200.0:
            raise ValueError("Defensive low-health threshold is invalid")
        if not 0.0 < self.health_drop_threshold <= 200.0:
            raise ValueError("Defensive health-drop threshold is invalid")
        if not math.isfinite(self.guard_bias) or self.guard_bias <= 0.0:
            raise ValueError("Defensive Guard bias must be positive and finite")
        if not math.isfinite(self.disengage_bias) or self.disengage_bias <= 0.0:
            raise ValueError("Defensive Disengage bias must be positive and finite")


@dataclass(frozen=True)
class ExplorerGovernorConfig:
    route_decisions: int
    route_bias: float
    flank_bias: float
    stall_repeat_decisions: int
    stall_recovery_decisions: int
    low_health_threshold: float
    max_consecutive_interventions: int

    def __post_init__(self) -> None:
        positive_ints = (
            self.route_decisions,
            self.stall_repeat_decisions,
            self.stall_recovery_decisions,
            self.max_consecutive_interventions,
        )
        if any(type(value) is not int or value <= 0 for value in positive_ints):
            raise ValueError("Explorer decision limits must be positive integers")
        if self.route_decisions > self.max_consecutive_interventions:
            raise ValueError("Route commitment exceeds the consecutive intervention limit")
        if not 0.0 < self.low_health_threshold <= 200.0:
            raise ValueError("Explorer low-health threshold is invalid")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (self.route_bias, self.flank_bias)
        ):
            raise ValueError("Explorer route biases must be positive and finite")


def _bias(**values: float) -> tuple[float, ...]:
    result = [0.0] * ACTION_COUNT
    for name, value in values.items():
        result[MacroAction[name]] = value
    return tuple(result)


class DefensiveGovernor:
    def __init__(self, config: DefensiveGovernorConfig) -> None:
        self.config = config
        self._episode_counter = 0
        self._state: DefensiveState = "base"
        self._remaining = 0
        self._elapsed = 0
        self._previous_health: float | None = None
        self._previous_score: int | None = None

    @property
    def episode_counter(self) -> int:
        return self._episode_counter

    def reset(self) -> None:
        self._episode_counter += 1
        self._state = "base"
        self._remaining = 0
        self._elapsed = 0
        self._previous_health = None
        self._previous_score = None

    def decide(self, context: PublicStyleContext) -> GovernorDecision:
        if self._previous_score is None:
            self._previous_score = context.own_score
            self._previous_health = context.health
            return self._decision("initial_observation", "base_warmup")

        score_rise = context.own_score > self._previous_score
        health_drop = max(0.0, float(self._previous_health) - context.health)
        low_health = context.health <= self.config.low_health_threshold
        self._previous_score = context.own_score
        self._previous_health = context.health

        if context.has_core:
            self._enter("base", 0)
            return self._decision("carrying", "objective_return_priority")

        if self._state == "recover":
            if self._remaining <= 0:
                self._enter("base", 0)
                return self._decision("cooldown_complete", "base_restored")
            decision = self._decision(
                "recover_cooldown",
                "exact_base_fallback",
                fallback="recover_base_only",
            )
            self._tick()
            return decision

        if self._state in ("guard", "disengage") and self._remaining <= 0:
            self._enter("recover", self.config.recover_decisions)
            decision = self._decision(
                "intervention_limit",
                "exact_base_fallback",
                fallback="recover_base_only",
            )
            self._tick()
            return decision

        if low_health or health_drop >= self.config.health_drop_threshold:
            if self._state != "disengage":
                self._enter("disengage", self.config.disengage_decisions)
            trigger = "low_health" if low_health else "health_drop"
            return self._active_decision(trigger)

        if score_rise:
            self._enter("guard", self.config.guard_decisions)
            return self._active_decision("own_score_rise")

        if self._state in ("guard", "disengage"):
            return self._active_decision("state_continue")

        return self._decision("no_public_trigger", "exact_base_fallback")

    def _enter(self, state: DefensiveState, remaining: int) -> None:
        self._state = state
        self._remaining = remaining
        self._elapsed = 0

    def _tick(self) -> None:
        self._remaining = max(0, self._remaining - 1)
        self._elapsed += 1

    def _active_decision(self, trigger: str) -> GovernorDecision:
        remaining = max(0, self._remaining - 1)
        if self._state == "guard":
            left = self._elapsed % 2 == 0
            magnitude = self.config.guard_bias
            bias = _bias(
                MOVE_FORWARD=-0.5 * magnitude,
                FORWARD_ATTACK=-0.25 * magnitude,
                **(
                    {
                        "TURN_LEFT": magnitude,
                        "STRAFE_LEFT": 0.7 * magnitude,
                        "TURN_LEFT_ATTACK": 0.5 * magnitude,
                    }
                    if left
                    else {
                        "TURN_RIGHT": magnitude,
                        "STRAFE_RIGHT": 0.7 * magnitude,
                        "TURN_RIGHT_ATTACK": 0.5 * magnitude,
                    }
                ),
            )
            reason = "guard_scan_left" if left else "guard_scan_right"
        else:
            left = self._elapsed % 2 == 0
            magnitude = self.config.disengage_bias
            bias = _bias(
                MOVE_BACKWARD=magnitude,
                MOVE_FORWARD=-magnitude,
                FORWARD_ATTACK=-magnitude,
                **(
                    {"STRAFE_LEFT": 0.8 * magnitude, "TURN_LEFT": 0.5 * magnitude}
                    if left
                    else {"STRAFE_RIGHT": 0.8 * magnitude, "TURN_RIGHT": 0.5 * magnitude}
                ),
            )
            reason = "disengage_left" if left else "disengage_right"
        decision = GovernorDecision(
            state=self._state,
            trigger=trigger,
            reason=reason,
            logit_bias=bias,
            max_remaining_interventions=remaining,
            fallback_condition="limit_health_or_carrying",
        )
        self._tick()
        return decision

    def _decision(
        self,
        trigger: str,
        reason: str,
        *,
        fallback: str = "inactive",
    ) -> GovernorDecision:
        return GovernorDecision(
            state=self._state,
            trigger=trigger,
            reason=reason,
            max_remaining_interventions=max(0, self._remaining),
            fallback_condition=fallback,
        )


class ExplorerGovernor:
    _MODES: tuple[ExplorerMode, ...] = ("upper", "lower", "flank")

    def __init__(self, config: ExplorerGovernorConfig) -> None:
        self.config = config
        self._episode_counter = 0
        self._episode_mode: ExplorerMode = "upper"
        self._next_mode_index = 0
        self._state: ExplorerState = "base"
        self._remaining = 0
        self._elapsed = 0
        self._previous_score: int | None = None
        self._previous_has_core: bool | None = None
        self._last_action: MacroAction | None = None
        self._repeat_actions = 0

    @property
    def episode_counter(self) -> int:
        return self._episode_counter

    @property
    def episode_mode(self) -> ExplorerMode:
        return self._episode_mode

    def reset(self) -> None:
        self._episode_mode = self._MODES[self._episode_counter % len(self._MODES)]
        self._next_mode_index = self._episode_counter % len(self._MODES)
        self._episode_counter += 1
        self._state = "base"
        self._remaining = 0
        self._elapsed = 0
        self._previous_score = None
        self._previous_has_core = None
        self._last_action = None
        self._repeat_actions = 0

    def decide(self, context: PublicStyleContext) -> GovernorDecision:
        self._update_repeat_count(context.previous_action)
        if self._previous_score is None:
            self._previous_score = context.own_score
            self._previous_has_core = context.has_core
            return self._decision("initial_observation", "base_warmup")

        score_rise = context.own_score > self._previous_score
        core_rise = context.has_core and not bool(self._previous_has_core)
        core_drop = not context.has_core and bool(self._previous_has_core)
        self._previous_score = context.own_score
        self._previous_has_core = context.has_core

        if score_rise:
            self._next_mode_index = (self._next_mode_index + 1) % len(self._MODES)
            self._episode_mode = self._MODES[self._next_mode_index]
            self._enter("stall_recovery", self.config.stall_recovery_decisions)
            return self._recovery_decision("own_score_rise")

        if self._state == "stall_recovery":
            if self._remaining <= 0:
                self._enter("base", 0)
                return self._decision("cooldown_complete", "base_restored")
            return self._recovery_decision("recovery_continue")

        if context.health <= self.config.low_health_threshold:
            self._enter("stall_recovery", self.config.stall_recovery_decisions)
            return self._recovery_decision("health_safety")

        if core_drop:
            self._enter("stall_recovery", self.config.stall_recovery_decisions)
            return self._recovery_decision("core_drop")

        if core_rise:
            self._enter("route_commit", self.config.route_decisions)
            return self._route_decision("core_rise")

        if self._state == "route_commit":
            if not context.has_core:
                self._enter("stall_recovery", self.config.stall_recovery_decisions)
                return self._recovery_decision("core_absent")
            if self._repeat_actions >= self.config.stall_repeat_decisions:
                self._enter("stall_recovery", self.config.stall_recovery_decisions)
                return self._recovery_decision("repeated_action_stall")
            if self._remaining <= 0:
                self._enter("stall_recovery", self.config.stall_recovery_decisions)
                return self._recovery_decision("intervention_limit")
            return self._route_decision("route_continue")

        return self._decision("no_public_trigger", "exact_base_fallback")

    def _update_repeat_count(self, action: MacroAction) -> None:
        if action == self._last_action:
            self._repeat_actions += 1
        else:
            self._last_action = action
            self._repeat_actions = 1

    def _enter(self, state: ExplorerState, remaining: int) -> None:
        self._state = state
        self._remaining = remaining
        self._elapsed = 0
        if state != "route_commit":
            self._repeat_actions = 0

    def _tick(self) -> None:
        self._remaining = max(0, self._remaining - 1)
        self._elapsed += 1

    def _route_decision(self, trigger: str) -> GovernorDecision:
        mode = self._episode_mode
        if mode == "upper":
            rhythm = (
                {"FORWARD_TURN_LEFT": 1.0, "TURN_LEFT": 0.5},
                {"STRAFE_LEFT": 1.0, "MOVE_FORWARD": 0.5},
                {"MOVE_FORWARD": 1.0},
            )
            magnitude = self.config.route_bias
        elif mode == "lower":
            rhythm = (
                {"FORWARD_TURN_RIGHT": 1.0, "TURN_RIGHT": 0.5},
                {"STRAFE_RIGHT": 1.0, "MOVE_FORWARD": 0.5},
                {"MOVE_FORWARD": 1.0},
            )
            magnitude = self.config.route_bias
        else:
            rhythm = (
                {"STRAFE_LEFT": 1.0, "FORWARD_TURN_LEFT": 0.5},
                {"MOVE_FORWARD": 1.0, "TURN_LEFT": 0.4},
                {"STRAFE_LEFT": 0.8, "MOVE_FORWARD": 0.5},
                {"STRAFE_RIGHT": 1.0, "FORWARD_TURN_RIGHT": 0.5},
                {"MOVE_FORWARD": 1.0, "TURN_RIGHT": 0.4},
                {"STRAFE_RIGHT": 0.8, "MOVE_FORWARD": 0.5},
            )
            magnitude = self.config.flank_bias
        phase = self._elapsed % len(rhythm)
        bias = _bias(**{name: scale * magnitude for name, scale in rhythm[phase].items()})
        decision = GovernorDecision(
            state=self._state,
            trigger=trigger,
            reason=f"{mode}_route_phase_{phase}",
            logit_bias=bias,
            max_remaining_interventions=max(0, self._remaining - 1),
            fallback_condition="limit_stall_health_score_or_drop",
            route_mode=mode,
        )
        self._tick()
        return decision

    def _recovery_decision(self, trigger: str) -> GovernorDecision:
        decision = self._decision(
            trigger,
            "exact_base_fallback",
            fallback="stall_recovery_base_only",
        )
        self._tick()
        return decision

    def _decision(
        self,
        trigger: str,
        reason: str,
        *,
        fallback: str = "inactive",
    ) -> GovernorDecision:
        return GovernorDecision(
            state=self._state,
            trigger=trigger,
            reason=reason,
            max_remaining_interventions=max(0, self._remaining),
            fallback_condition=fallback,
            route_mode=self._episode_mode if self._state == "route_commit" else None,
        )
