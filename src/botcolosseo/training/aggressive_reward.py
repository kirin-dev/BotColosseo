from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_protocol import DuelEvent, DuelEventType
from botcolosseo.envs.duel_types import DuelPrivilegedState

_ATTACKS = frozenset(
    {
        MacroAction.ATTACK,
        MacroAction.FORWARD_ATTACK,
        MacroAction.TURN_LEFT_ATTACK,
        MacroAction.TURN_RIGHT_ATTACK,
    }
)


@dataclass(frozen=True)
class AggressiveRewardConfig:
    valid_hit: float = 0.10
    engagement_initiation: float = 0.15
    forward_hit: float = 0.05
    invalid_attack: float = -0.02
    objective_chase: float = -0.03
    valid_hit_cap: int = 20
    initiation_cap: int = 8
    forward_hit_cap: int = 12
    invalid_attack_cap: int = 30
    objective_chase_cap: int = 15
    initiation_cooldown: int = 12

    def __post_init__(self) -> None:
        caps = (
            self.valid_hit_cap,
            self.initiation_cap,
            self.forward_hit_cap,
            self.invalid_attack_cap,
            self.objective_chase_cap,
        )
        if any(value < 0 for value in caps) or self.initiation_cooldown < 0:
            raise ValueError("Aggressive reward caps and cooldown must be nonnegative")


@dataclass(frozen=True)
class AggressiveReward:
    total: float
    components: dict[str, float]


class AggressiveRewardLedger:
    """Opportunity-conditioned aggression shaping using public actions/events only."""

    def __init__(
        self,
        config: AggressiveRewardConfig,
        *,
        learner_side: str,
        scale: float = 1.0,
    ) -> None:
        if learner_side not in ("host", "opponent"):
            raise ValueError("learner_side must be host or opponent")
        self.config = config
        self.learner_side = learner_side
        if scale < 0:
            raise ValueError("Aggressive reward scale must be nonnegative")
        self.scale = scale
        self.reset()

    def reset(self) -> None:
        self._counts: Counter[str] = Counter()
        self._steps_since_hit = self.config.initiation_cooldown + 1

    def _add(
        self, components: dict[str, float], name: str, value: float, cap: int
    ) -> None:
        if self._counts[name] >= cap:
            return
        self._counts[name] += 1
        components[name] = components.get(name, 0.0) + value

    def apply(
        self,
        action: MacroAction,
        events: tuple[DuelEvent, ...],
        *,
        has_core: bool,
        state_before: DuelPrivilegedState | None = None,
        state_after: DuelPrivilegedState | None = None,
    ) -> AggressiveReward:
        del state_before, state_after
        action = MacroAction(action)
        components: dict[str, float] = {}
        valid_hit = any(
            event.side == self.learner_side
            and event.type is DuelEventType.VALID_HIT
            for event in events
        )
        if valid_hit:
            self._add(
                components, "valid_hit", self.config.valid_hit, self.config.valid_hit_cap
            )
            if self._steps_since_hit > self.config.initiation_cooldown:
                self._add(
                    components,
                    "engagement_initiation",
                    self.config.engagement_initiation,
                    self.config.initiation_cap,
                )
            if action is MacroAction.FORWARD_ATTACK:
                self._add(
                    components,
                    "forward_hit",
                    self.config.forward_hit,
                    self.config.forward_hit_cap,
                )
            self._steps_since_hit = 0
        else:
            self._steps_since_hit += 1
            if action in _ATTACKS:
                self._add(
                    components,
                    "invalid_attack",
                    self.config.invalid_attack,
                    self.config.invalid_attack_cap,
                )
                if has_core:
                    self._add(
                        components,
                        "objective_chase",
                        self.config.objective_chase,
                        self.config.objective_chase_cap,
                    )
        scaled = {name: value * self.scale for name, value in components.items()}
        return AggressiveReward(sum(scaled.values()), scaled)
