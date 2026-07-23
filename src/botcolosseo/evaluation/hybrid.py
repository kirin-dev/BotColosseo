from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from botcolosseo.agents.style_governor import ExplorerMode, GovernorTelemetry
from botcolosseo.envs.actions import MacroAction
from botcolosseo.evaluation.defensive import (
    DefensiveEpisodeRecord,
    DefensiveEvaluationSummary,
)
from botcolosseo.evaluation.explorer import (
    ExplorerEpisodeRecord,
    ExplorerEvaluationSummary,
)

HybridStyle = Literal["defensive", "explorer"]
EpisodeRecord = DefensiveEpisodeRecord | ExplorerEpisodeRecord
LegacySummary = DefensiveEvaluationSummary | ExplorerEvaluationSummary


@dataclass(frozen=True)
class HybridTelemetryRow:
    style: HybridStyle
    opponent: str
    pair_index: int
    learner_side: str
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

    @classmethod
    def from_policy(
        cls,
        *,
        style: HybridStyle,
        opponent: str,
        pair_index: int,
        learner_side: str,
        telemetry: GovernorTelemetry,
    ) -> HybridTelemetryRow:
        return cls(
            style=style,
            opponent=opponent,
            pair_index=pair_index,
            learner_side=learner_side,
            **asdict(telemetry),
        )

    def __post_init__(self) -> None:
        if self.style not in ("defensive", "explorer"):
            raise ValueError("Unknown hybrid telemetry style")
        if self.learner_side not in ("host", "opponent"):
            raise ValueError("Hybrid telemetry learner side is invalid")
        if self.pair_index < 0 or self.decision_index < 0:
            raise ValueError("Hybrid telemetry counters must be nonnegative")
        if not self.opponent or not self.state or not self.trigger or not self.reason:
            raise ValueError("Hybrid telemetry labels must be non-empty")
        MacroAction(self.base_action)
        MacroAction(self.final_action)
        if not self.intervened and self.final_action != self.base_action:
            raise ValueError("Non-intervened hybrid action must exactly match Base")
        if self.used_override and not self.intervened:
            raise ValueError("Hybrid override must count as an intervention")
        if self.style == "defensive" and self.route_mode is not None:
            raise ValueError("Defensive telemetry cannot contain an Explorer mode")

    @property
    def episode_key(self) -> tuple[str, int, str]:
        return self.opponent, self.pair_index, self.learner_side

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HybridProductSummary:
    style: HybridStyle
    complete: bool
    passed: bool
    episodes: int
    expected_episodes: int
    protocol_inconsistencies: int
    skill_retention: float
    per_opponent_retention: dict[str, float]
    decisions: int
    interventions: int
    intervention_rate: float
    overrides: int
    fallbacks: int
    max_consecutive_interventions: int
    state_occupancy: dict[str, int]
    trigger_counts: dict[str, int]
    reason_counts: dict[str, int]
    route_mode_counts: dict[str, int]
    route_action_signature_distance: float | None
    gates: dict[str, bool]
    legacy_diagnostic_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _maximum_consecutive(rows: Sequence[HybridTelemetryRow]) -> int:
    grouped: dict[tuple[str, int, str], list[HybridTelemetryRow]] = defaultdict(list)
    for row in rows:
        grouped[row.episode_key].append(row)
    maximum = 0
    for episode_rows in grouped.values():
        consecutive = 0
        for row in sorted(episode_rows, key=lambda item: item.decision_index):
            consecutive = consecutive + 1 if row.intervened else 0
            maximum = max(maximum, consecutive)
    return maximum


def _route_signature_distance(rows: Sequence[HybridTelemetryRow]) -> float | None:
    modes = ("upper", "lower", "flank")
    counts = {
        mode: Counter(
            row.final_action
            for row in rows
            if row.intervened and row.route_mode == mode
        )
        for mode in modes
    }
    if any(not counts[mode] for mode in modes):
        return None

    def distribution(mode: str) -> tuple[float, ...]:
        total = sum(counts[mode].values())
        return tuple(counts[mode][action] / total for action in MacroAction)

    distributions = {mode: distribution(mode) for mode in modes}
    distances = [
        0.5
        * sum(
            abs(left - right)
            for left, right in zip(distributions[a], distributions[b], strict=True)
        )
        for index, a in enumerate(modes)
        for b in modes[index + 1 :]
    ]
    return min(distances)


def evaluate_hybrid_product(
    *,
    style: HybridStyle,
    records: Sequence[EpisodeRecord],
    telemetry: Sequence[HybridTelemetryRow],
    legacy_summary: LegacySummary,
    max_consecutive_interventions: int,
) -> HybridProductSummary:
    if style not in ("defensive", "explorer"):
        raise ValueError("Unknown hybrid style")
    if max_consecutive_interventions <= 0:
        raise ValueError("Hybrid intervention limit must be positive")
    style_records = [row for row in records if row.policy == style]
    if any(row.style != style for row in telemetry):
        raise ValueError("Hybrid telemetry style does not match evaluation")
    expected_decisions = sum(row.decisions for row in style_records)
    episode_keys = {
        (row.opponent, row.pair_index, row.learner_side) for row in style_records
    }
    telemetry_keys = {row.episode_key for row in telemetry}
    decision_counts = Counter(row.episode_key for row in telemetry)
    complete_telemetry = (
        len(telemetry) == expected_decisions
        and telemetry_keys == episode_keys
        and all(
            decision_counts[(row.opponent, row.pair_index, row.learner_side)]
            == row.decisions
            for row in style_records
        )
    )
    interventions = sum(row.intervened for row in telemetry)
    overrides = sum(row.used_override for row in telemetry)
    fallbacks = sum(not row.intervened for row in telemetry)
    maximum = _maximum_consecutive(telemetry)
    state_occupancy = dict(sorted(Counter(row.state for row in telemetry).items()))
    trigger_counts = dict(sorted(Counter(row.trigger for row in telemetry).items()))
    reason_counts = dict(sorted(Counter(row.reason for row in telemetry).items()))
    route_mode_counts = dict(
        sorted(Counter(row.route_mode for row in telemetry if row.route_mode).items())
    )
    signature_distance = (
        _route_signature_distance(telemetry) if style == "explorer" else None
    )
    exact_base_fallback = all(
        row.intervened or row.final_action == row.base_action for row in telemetry
    )
    coverage = (
        state_occupancy.get("guard", 0) > 0
        and state_occupancy.get("disengage", 0) > 0
        and state_occupancy.get("recover", 0) > 0
        if style == "defensive"
        else all(route_mode_counts.get(mode, 0) > 0 for mode in ("upper", "lower", "flank"))
        and signature_distance is not None
        and signature_distance >= 0.05
    )
    gates = {
        "complete": legacy_summary.complete,
        "protocol_clean": legacy_summary.protocol_inconsistencies == 0,
        "skill_retention": legacy_summary.skill_retention >= 0.85,
        "per_opponent_retention": all(
            value >= 0.75 for value in legacy_summary.per_opponent_retention.values()
        ),
        "telemetry_complete": complete_telemetry,
        "intervention_nonzero": interventions > 0,
        "intervention_bounded": maximum <= max_consecutive_interventions,
        "exact_base_fallback": exact_base_fallback,
        "required_style_coverage": coverage,
    }
    return HybridProductSummary(
        style=style,
        complete=legacy_summary.complete and complete_telemetry,
        passed=all(gates.values()),
        episodes=len(records),
        expected_episodes=legacy_summary.expected_episodes,
        protocol_inconsistencies=legacy_summary.protocol_inconsistencies,
        skill_retention=legacy_summary.skill_retention,
        per_opponent_retention=legacy_summary.per_opponent_retention,
        decisions=len(telemetry),
        interventions=interventions,
        intervention_rate=interventions / len(telemetry) if telemetry else 0.0,
        overrides=overrides,
        fallbacks=fallbacks,
        max_consecutive_interventions=maximum,
        state_occupancy=state_occupancy,
        trigger_counts=trigger_counts,
        reason_counts=reason_counts,
        route_mode_counts=route_mode_counts,
        route_action_signature_distance=signature_distance,
        gates=gates,
        legacy_diagnostic_passed=legacy_summary.passed,
    )
