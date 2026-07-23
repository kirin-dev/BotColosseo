from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.defensive_signals import (
    defensive_risk,
    in_defensive_half,
    in_protective_zone,
    learner_carrier_id,
    opponent_carrier_id,
    unnecessary_guard,
)
from botcolosseo.envs.duel_protocol import DuelEvent, DuelEventType
from botcolosseo.envs.duel_types import DuelPrivilegedState


@dataclass(frozen=True)
class DefensiveRewardConfig:
    risk_presence: float = 0.01
    protective_entry: float = 0.05
    carrier_denial: float = 0.20
    defensive_recovery: float = 0.15
    risk_resolution: float = 0.10
    unnecessary_guard: float = -0.02
    risk_concession: float = -0.20
    risk_presence_cap: int = 40
    protective_entry_cap: int = 8
    carrier_denial_cap: int = 6
    defensive_recovery_cap: int = 6
    risk_resolution_cap: int = 6
    unnecessary_guard_cap: int = 30
    risk_concession_cap: int = 5

    def __post_init__(self) -> None:
        caps = (
            self.risk_presence_cap,
            self.protective_entry_cap,
            self.carrier_denial_cap,
            self.defensive_recovery_cap,
            self.risk_resolution_cap,
            self.unnecessary_guard_cap,
            self.risk_concession_cap,
        )
        if any(type(value) is not int or value < 0 for value in caps):
            raise ValueError("Defensive reward caps must be nonnegative integers")


@dataclass(frozen=True)
class DefensiveReward:
    total: float
    components: dict[str, float]


class DefensiveRewardLedger:
    def __init__(
        self,
        config: DefensiveRewardConfig,
        *,
        learner_side: str,
        scale: float = 1.0,
    ) -> None:
        if learner_side not in ("host", "opponent"):
            raise ValueError("learner_side must be host or opponent")
        if scale < 0:
            raise ValueError("Defensive reward scale must be nonnegative")
        self.config = config
        self.learner_side = learner_side
        self.scale = scale
        self.reset()

    def reset(self) -> None:
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

    def apply(
        self,
        action: MacroAction,
        events: tuple[DuelEvent, ...],
        *,
        has_core: bool,
        state_before: DuelPrivilegedState | None = None,
        state_after: DuelPrivilegedState | None = None,
    ) -> DefensiveReward:
        del action, has_core
        if state_before is None or state_after is None:
            raise ValueError("Defensive reward requires before/after privileged state")
        side = self.learner_side
        opponent_side = "opponent" if side == "host" else "host"
        risk_before = defensive_risk(state_before, side)
        risk_after = defensive_risk(state_after, side)
        before_zone = in_protective_zone(state_before, side)
        after_zone = in_protective_zone(state_after, side)
        opponent_scored = any(
            event.side == opponent_side and event.type is DuelEventType.SCORE
            for event in events
        )
        learner_hit = any(
            event.side == side and event.type is DuelEventType.VALID_HIT
            for event in events
        )
        components: dict[str, float] = {}
        if risk_before and after_zone:
            self._add(
                components,
                "risk_presence",
                self.config.risk_presence,
                self.config.risk_presence_cap,
            )
            if not before_zone:
                self._add(
                    components,
                    "protective_entry",
                    self.config.protective_entry,
                    self.config.protective_entry_cap,
                )
        if (
            state_before.carrier == opponent_carrier_id(side)
            and state_after.carrier != opponent_carrier_id(side)
            and learner_hit
        ):
            self._add(
                components,
                "carrier_denial",
                self.config.carrier_denial,
                self.config.carrier_denial_cap,
            )
        if (
            state_before.carrier == 0
            and in_defensive_half(state_before.core_x, side)
            and state_after.carrier == learner_carrier_id(side)
        ):
            self._add(
                components,
                "defensive_recovery",
                self.config.defensive_recovery,
                self.config.defensive_recovery_cap,
            )
        if risk_before and not risk_after and not opponent_scored:
            self._add(
                components,
                "risk_resolution",
                self.config.risk_resolution,
                self.config.risk_resolution_cap,
            )
        if not risk_before and unnecessary_guard(state_after, side):
            self._add(
                components,
                "unnecessary_guard",
                self.config.unnecessary_guard,
                self.config.unnecessary_guard_cap,
            )
        if risk_before and opponent_scored:
            self._add(
                components,
                "risk_concession",
                self.config.risk_concession,
                self.config.risk_concession_cap,
            )
        scaled = {name: value * self.scale for name, value in components.items()}
        return DefensiveReward(sum(scaled.values()), scaled)
