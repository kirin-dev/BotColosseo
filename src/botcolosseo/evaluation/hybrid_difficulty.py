from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from botcolosseo.agents.difficulty import DifficultyProfile
from botcolosseo.agents.hybrid_difficulty import HybridExecutionTrace
from botcolosseo.agents.style_governor import GovernorTelemetry
from botcolosseo.envs.actions import MacroAction
from botcolosseo.evaluation.native_style_difficulty import (
    NativeStyle,
    NativeStyleDifficultyRecord,
)
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS

HYBRID_DIFFICULTIES = ("easy", "normal")


@dataclass(frozen=True)
class HybridDifficultyGovernorRow:
    style: NativeStyle
    difficulty: Literal["easy", "normal"]
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
    route_mode: str | None

    @classmethod
    def from_telemetry(
        cls,
        *,
        style: NativeStyle,
        difficulty: str,
        opponent: str,
        pair_index: int,
        learner_side: str,
        telemetry: GovernorTelemetry,
    ) -> HybridDifficultyGovernorRow:
        return cls(
            style=style,
            difficulty=_difficulty(difficulty),
            opponent=opponent,
            pair_index=pair_index,
            learner_side=learner_side,
            **asdict(telemetry),
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> HybridDifficultyGovernorRow:
        values = dict(payload)
        values["base_action"] = MacroAction(values["base_action"])
        values["final_action"] = MacroAction(values["final_action"])
        return cls(**values)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        _row_identity(self)
        if self.decision_index < 0:
            raise ValueError("Hybrid difficulty governor decision is invalid")
        if not self.intervened and self.final_action != self.base_action:
            raise ValueError("Hybrid difficulty non-intervention changed Base")
        if self.used_override and not self.intervened:
            raise ValueError("Hybrid difficulty override is not an intervention")
        if self.style == "defensive" and self.route_mode is not None:
            raise ValueError("Defensive hybrid difficulty row has a route mode")

    @property
    def episode_key(self) -> tuple[str, str, int, str]:
        return self.difficulty, self.opponent, self.pair_index, self.learner_side

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HybridDifficultyExecutionRow:
    style: NativeStyle
    difficulty: Literal["easy", "normal"]
    opponent: str
    pair_index: int
    learner_side: str
    decision_index: int
    policy_updated: bool
    proposed_action: MacroAction
    emitted_action: MacroAction
    source_decision_index: int | None
    base_action: MacroAction | None
    state: str
    trigger: str
    reason: str
    intervened: bool
    used_override: bool
    fallback_condition: str
    route_mode: str | None
    warmup: bool

    @classmethod
    def from_trace(
        cls,
        *,
        style: NativeStyle,
        difficulty: str,
        opponent: str,
        pair_index: int,
        learner_side: str,
        trace: HybridExecutionTrace,
    ) -> HybridDifficultyExecutionRow:
        return cls(
            style=style,
            difficulty=_difficulty(difficulty),
            opponent=opponent,
            pair_index=pair_index,
            learner_side=learner_side,
            **asdict(trace),
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> HybridDifficultyExecutionRow:
        values = dict(payload)
        values["proposed_action"] = MacroAction(values["proposed_action"])
        values["emitted_action"] = MacroAction(values["emitted_action"])
        if values.get("base_action") is not None:
            values["base_action"] = MacroAction(values["base_action"])
        return cls(**values)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        _row_identity(self)
        if self.decision_index < 0:
            raise ValueError("Hybrid difficulty execution decision is invalid")
        if self.warmup != (self.source_decision_index is None):
            raise ValueError("Hybrid difficulty warm-up provenance is invalid")
        if self.warmup:
            if (
                self.emitted_action is not MacroAction.IDLE
                or self.base_action is not None
                or self.intervened
                or self.used_override
                or self.state != "warmup"
            ):
                raise ValueError("Hybrid difficulty warm-up row is invalid")
        elif self.base_action is None:
            raise ValueError("Hybrid difficulty execution source is incomplete")

    @property
    def episode_key(self) -> tuple[str, str, int, str]:
        return self.difficulty, self.opponent, self.pair_index, self.learner_side

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_hybrid_difficulty_extension(
    records: Sequence[NativeStyleDifficultyRecord],
    governor_rows: Sequence[HybridDifficultyGovernorRow],
    execution_rows: Sequence[HybridDifficultyExecutionRow],
    *,
    style: NativeStyle,
    profiles: Mapping[str, DifficultyProfile],
    max_consecutive_interventions: int,
    expected_pairs_per_opponent: int,
    expected_scenario_hash: str,
) -> dict[str, object]:
    if (
        style not in ("defensive", "explorer")
        or expected_pairs_per_opponent <= 0
        or max_consecutive_interventions <= 0
        or set(profiles) != set(HYBRID_DIFFICULTIES)
    ):
        raise ValueError("Hybrid difficulty evaluation settings are invalid")
    expected_per_tier = (
        len(DUEL_OPPONENTS) * expected_pairs_per_opponent * 2
    )
    expected = expected_per_tier * len(HYBRID_DIFFICULTIES)
    record_keys = [
        (
            row.difficulty,
            row.episode.opponent,
            row.episode.pair_index,
            row.episode.learner_side,
        )
        for row in records
    ]
    schedule_complete = (
        len(records) == expected
        and len(record_keys) == len(set(record_keys))
        and all(
            row.difficulty in HYBRID_DIFFICULTIES
            and row.episode.policy == style
            and row.episode.scenario_hash == expected_scenario_hash
            for row in records
        )
        and all(
            _case_count(records, difficulty, expected_pairs_per_opponent)
            for difficulty in HYBRID_DIFFICULTIES
        )
    )
    protocol_clean = len(record_keys) == len(set(record_keys)) and all(
        not row.episode.protocol_inconsistent
        and not row.episode.action_tic_inconsistent
        and not row.episode.score_event_inconsistent
        and row.episode.peer_tic_lag_max == 0
        and row.episode.terminated
        and not row.episode.truncated
        for row in records
    )
    tier_payloads = {}
    tier_passed = {}
    for difficulty in HYBRID_DIFFICULTIES:
        tier_records = [row for row in records if row.difficulty == difficulty]
        tier_governor = [
            row for row in governor_rows if row.difficulty == difficulty
        ]
        tier_execution = [
            row for row in execution_rows if row.difficulty == difficulty
        ]
        payload = _tier(
            tier_records,
            tier_governor,
            tier_execution,
            style=style,
            profile=profiles[difficulty],
            max_consecutive_interventions=max_consecutive_interventions,
        )
        tier_payloads[difficulty] = payload
        tier_passed[difficulty] = payload["passed"]
    all_evidence_keys = {
        row.episode_key for row in (*governor_rows, *execution_rows)
    }
    gates = {
        "complete": schedule_complete,
        "protocol_clean": protocol_clean,
        "evidence_case_identity": all_evidence_keys == set(record_keys),
        "style_mechanism_each_tier": all(tier_passed.values()),
    }
    return {
        "schema_version": 1,
        "stage": f"m5-hybrid-{style}-difficulty-extension",
        "style": style,
        "complete": schedule_complete,
        "passed": all(gates.values()),
        "episodes": len(records),
        "expected_episodes": expected,
        "protocol_inconsistencies": sum(
            int(
                row.episode.protocol_inconsistent
                or row.episode.action_tic_inconsistent
                or row.episode.score_event_inconsistent
                or row.episode.peer_tic_lag_max != 0
            )
            for row in records
        )
        + len(record_keys)
        - len(set(record_keys)),
        "environment_retries": sum(
            row.episode.environment_attempts - 1 for row in records
        ),
        "governor_decisions": len(governor_rows),
        "executed_decisions": len(execution_rows),
        "gates": gates,
        "tiers": tier_payloads,
        "test_cases_accessed": False,
    }


def _tier(
    records: Sequence[NativeStyleDifficultyRecord],
    governor_rows: Sequence[HybridDifficultyGovernorRow],
    execution_rows: Sequence[HybridDifficultyExecutionRow],
    *,
    style: NativeStyle,
    profile: DifficultyProfile,
    max_consecutive_interventions: int,
) -> dict[str, object]:
    expected_decisions = sum(row.episode.decisions for row in records)
    expected_updates = sum(
        math.ceil(row.episode.decisions / profile.policy_update_interval)
        for row in records
    )
    record_decisions = {
        (
            row.difficulty,
            row.episode.opponent,
            row.episode.pair_index,
            row.episode.learner_side,
        ): row.episode.decisions
        for row in records
    }
    governor_counts = Counter(row.episode_key for row in governor_rows)
    execution_counts = Counter(row.episode_key for row in execution_rows)
    evidence_complete = (
        len(governor_rows) == expected_updates
        and len(execution_rows) == expected_decisions
        and set(governor_counts) == set(record_decisions)
        and set(execution_counts) == set(record_decisions)
        and all(
            execution_counts[key] == decisions
            and governor_counts[key]
            == math.ceil(decisions / profile.policy_update_interval)
            for key, decisions in record_decisions.items()
        )
        and _indices_complete(governor_rows)
        and _indices_complete(execution_rows)
    )
    update_accounting = (
        sum(row.policy_updated for row in execution_rows) == expected_updates
    )
    warmup_accounting = all(
        sum(row.warmup for row in execution_rows if row.episode_key == key)
        == min(profile.reaction_delay, decisions)
        for key, decisions in record_decisions.items()
    )
    proposed_maximum = _maximum_consecutive(
        governor_rows,
        lambda row: row.intervened,
    )
    executed_maximum = _maximum_consecutive(
        execution_rows,
        lambda row: row.intervened,
    )
    exact_fallback = all(
        row.intervened or row.final_action == row.base_action
        for row in governor_rows
    )
    executed = [row for row in execution_rows if not row.warmup]
    states = dict(sorted(Counter(row.state for row in executed).items()))
    modes = dict(
        sorted(Counter(row.route_mode for row in executed if row.route_mode).items())
    )
    signature = _route_signature(executed) if style == "explorer" else None
    coverage = (
        all(states.get(state, 0) > 0 for state in ("guard", "disengage", "recover"))
        if style == "defensive"
        else all(modes.get(mode, 0) > 0 for mode in ("upper", "lower", "flank"))
        and signature is not None
        and signature >= 0.05
    )
    gates = {
        "evidence_complete": evidence_complete,
        "update_accounting": update_accounting,
        "warmup_accounting": warmup_accounting,
        "intervention_nonzero": any(row.intervened for row in executed),
        "exact_base_fallback": exact_fallback,
        "proposed_intervention_bounded": (
            proposed_maximum <= max_consecutive_interventions
        ),
        "executed_intervention_bounded": (
            executed_maximum
            <= max_consecutive_interventions * profile.policy_update_interval
        ),
        "required_style_coverage": coverage,
    }
    return {
        "passed": all(gates.values()),
        "episodes": len(records),
        "governor_decisions": len(governor_rows),
        "executed_decisions": len(execution_rows),
        "interventions": sum(row.intervened for row in executed),
        "intervention_rate": (
            sum(row.intervened for row in executed) / len(executed)
            if executed
            else 0.0
        ),
        "max_consecutive_proposed_interventions": proposed_maximum,
        "max_consecutive_executed_interventions": executed_maximum,
        "state_occupancy": states,
        "route_mode_counts": modes,
        "executed_route_signature_distance": signature,
        "gates": gates,
    }


def _case_count(
    records: Sequence[NativeStyleDifficultyRecord],
    difficulty: str,
    pairs_per_opponent: int,
) -> bool:
    selected = [row for row in records if row.difficulty == difficulty]
    expected = len(DUEL_OPPONENTS) * pairs_per_opponent * 2
    return len(selected) == expected and Counter(
        row.episode.opponent for row in selected
    ) == Counter(
        {opponent: pairs_per_opponent * 2 for opponent in DUEL_OPPONENTS}
    )


def _indices_complete(
    rows: Sequence[HybridDifficultyGovernorRow]
    | Sequence[HybridDifficultyExecutionRow],
) -> bool:
    grouped: dict[tuple[str, str, int, str], list[int]] = defaultdict(list)
    for row in rows:
        grouped[row.episode_key].append(row.decision_index)
    return all(sorted(indices) == list(range(len(indices))) for indices in grouped.values())


def _maximum_consecutive(
    rows: Sequence[HybridDifficultyGovernorRow]
    | Sequence[HybridDifficultyExecutionRow],
    predicate,
) -> int:
    grouped: dict[
        tuple[str, str, int, str],
        list[HybridDifficultyGovernorRow | HybridDifficultyExecutionRow],
    ] = defaultdict(list)
    for row in rows:
        grouped[row.episode_key].append(row)
    maximum = 0
    for episode_rows in grouped.values():
        count = 0
        for row in sorted(episode_rows, key=lambda item: item.decision_index):
            count = count + 1 if predicate(row) else 0
            maximum = max(maximum, count)
    return maximum


def _route_signature(
    rows: Sequence[HybridDifficultyExecutionRow],
) -> float | None:
    modes = ("upper", "lower", "flank")
    counts = {
        mode: Counter(
            row.emitted_action
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
    return min(
        0.5
        * sum(
            abs(left - right)
            for left, right in zip(
                distributions[first],
                distributions[second],
                strict=True,
            )
        )
        for index, first in enumerate(modes)
        for second in modes[index + 1 :]
    )


def _difficulty(value: str) -> Literal["easy", "normal"]:
    if value not in HYBRID_DIFFICULTIES:
        raise ValueError("Hybrid difficulty row requires Easy or Normal")
    return value  # type: ignore[return-value]


def _row_identity(
    row: HybridDifficultyGovernorRow | HybridDifficultyExecutionRow,
) -> None:
    if (
        row.style not in ("defensive", "explorer")
        or row.difficulty not in HYBRID_DIFFICULTIES
        or row.opponent not in DUEL_OPPONENTS
        or row.pair_index < 0
        or row.learner_side not in ("host", "opponent")
    ):
        raise ValueError("Hybrid difficulty row identity is invalid")
