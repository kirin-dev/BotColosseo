from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from botcolosseo.agents.duel_teachers import EXPLORER_ROUTE_CYCLE
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.defensive_signals import learner_carrier_id
from botcolosseo.envs.duel_protocol import DuelEvent, DuelEventType
from botcolosseo.envs.duel_types import DuelPrivilegedState

_ROUTE_REGIONS = {
    "direct_upper": {"upper_route"},
    "direct_lower": {"lower_route"},
    "flank": {"flank_west", "flank_east"},
}
_FLANK_REGIONS = _ROUTE_REGIONS["flank"]


@dataclass(frozen=True)
class ExplorerRewardConfig:
    target_region: float = 0.05
    target_route_score: float = 0.25
    novel_carry_region: float = 0.01
    carry_stall: float = -0.01
    distinct_route_completion: float = 0.0
    target_region_cap: int = 12
    target_route_score_cap: int = 5
    novel_carry_region_cap: int = 24
    carry_stall_cap: int = 30
    distinct_route_completion_cap: int = 0
    stall_decisions: int = 12

    def __post_init__(self) -> None:
        caps = (
            self.target_region_cap,
            self.target_route_score_cap,
            self.novel_carry_region_cap,
            self.carry_stall_cap,
            self.distinct_route_completion_cap,
        )
        if (
            any(type(value) is not int or value < 0 for value in caps)
            or type(self.stall_decisions) is not int
            or self.stall_decisions < 0
        ):
            raise ValueError("Explorer reward caps and stall limit must be nonnegative")


@dataclass(frozen=True)
class ExplorerReward:
    total: float
    components: dict[str, float]


class ExplorerRewardLedger:
    def __init__(
        self,
        config: ExplorerRewardConfig,
        *,
        learner_side: str,
        scale: float = 1.0,
    ) -> None:
        if learner_side not in ("host", "opponent"):
            raise ValueError("learner_side must be host or opponent")
        if scale < 0:
            raise ValueError("Explorer reward scale must be nonnegative")
        self.config = config
        self.learner_side = learner_side
        self.scale = scale
        self.reset()

    def reset(self) -> None:
        self._counts: Counter[str] = Counter()
        self._initial_score: int | None = None
        self._carry_regions: set[str] = set()
        self._target_regions_rewarded: set[str] = set()
        self._last_region: str | None = None
        self._same_region_decisions = 0
        self._route_mode: int | None = None
        self._completed_routes: set[str] = set()

    def set_route_mode(self, route_mode: int) -> None:
        if not 0 <= route_mode < len(EXPLORER_ROUTE_CYCLE):
            raise ValueError("Invalid Explorer reward route mode")
        self._route_mode = route_mode

    def _score(self, state: DuelPrivilegedState) -> int:
        return (
            state.host_score
            if self.learner_side == "host"
            else state.opponent_score
        )

    def _region(self, state: DuelPrivilegedState) -> str | None:
        return (
            state.host_region
            if self.learner_side == "host"
            else state.opponent_region
        )

    def _target(self, state: DuelPrivilegedState) -> str:
        if self._route_mode is not None:
            return EXPLORER_ROUTE_CYCLE[self._route_mode]
        if self._initial_score is None:
            self._initial_score = self._score(state)
        progress = self._score(state) - self._initial_score
        return EXPLORER_ROUTE_CYCLE[progress % len(EXPLORER_ROUTE_CYCLE)]

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

    def _clear_carry(self) -> None:
        self._carry_regions.clear()
        self._target_regions_rewarded.clear()
        self._last_region = None
        self._same_region_decisions = 0

    @staticmethod
    def _matches_target(target: str, regions: set[str]) -> bool:
        if target == "flank":
            return _FLANK_REGIONS.issubset(regions)
        if not regions.isdisjoint(_FLANK_REGIONS):
            return False
        if target == "direct_upper":
            return "upper_route" in regions
        return "lower_route" in regions and "upper_route" not in regions

    def apply(
        self,
        action: MacroAction,
        events: tuple[DuelEvent, ...],
        *,
        has_core: bool,
        state_before: DuelPrivilegedState | None = None,
        state_after: DuelPrivilegedState | None = None,
    ) -> ExplorerReward:
        del action, has_core
        if state_before is None or state_after is None:
            raise ValueError("Explorer reward requires before/after privileged state")
        target = self._target(state_before)
        learner_id = learner_carrier_id(self.learner_side)
        components: dict[str, float] = {}
        if (
            state_after.carrier == learner_id
            and state_before.carrier != learner_id
        ):
            self._clear_carry()
        if state_after.carrier == learner_id:
            region = self._region(state_after)
            if region is not None:
                if region not in self._carry_regions:
                    self._carry_regions.add(region)
                    self._add(
                        components,
                        "novel_carry_region",
                        self.config.novel_carry_region,
                        self.config.novel_carry_region_cap,
                    )
                if (
                    region in _ROUTE_REGIONS[target]
                    and region not in self._target_regions_rewarded
                ):
                    self._target_regions_rewarded.add(region)
                    self._add(
                        components,
                        "target_region",
                        self.config.target_region,
                        self.config.target_region_cap,
                    )
                if region == self._last_region:
                    self._same_region_decisions += 1
                else:
                    self._last_region = region
                    self._same_region_decisions = 1
                if self._same_region_decisions > self.config.stall_decisions:
                    self._add(
                        components,
                        "carry_stall",
                        self.config.carry_stall,
                        self.config.carry_stall_cap,
                    )
        learner_scored = any(
            event.side == self.learner_side
            and event.type is DuelEventType.SCORE
            for event in events
        )
        if learner_scored and self._matches_target(target, self._carry_regions):
            self._add(
                components,
                "target_route_score",
                self.config.target_route_score,
                self.config.target_route_score_cap,
            )
            if target not in self._completed_routes:
                self._completed_routes.add(target)
                self._add(
                    components,
                    "distinct_route_completion",
                    self.config.distinct_route_completion,
                    self.config.distinct_route_completion_cap,
                )
        if state_after.carrier != learner_id:
            self._clear_carry()
        scaled = {name: value * self.scale for name, value in components.items()}
        return ExplorerReward(sum(scaled.values()), scaled)
