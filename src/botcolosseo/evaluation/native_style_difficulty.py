from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

from botcolosseo.agents.difficulty import DIFFICULTIES
from botcolosseo.evaluation.defensive import (
    DEFENSIVE_POLICIES,
    DefensiveEpisodeRecord,
    evaluate_defensive_records,
)
from botcolosseo.evaluation.difficulty import MONOTONIC_TOLERANCE
from botcolosseo.evaluation.explorer import (
    EXPLORER_POLICIES,
    ExplorerEpisodeRecord,
    evaluate_explorer_records,
)
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS

NativeStyle = Literal["defensive", "explorer"]
NativeEpisode = DefensiveEpisodeRecord | ExplorerEpisodeRecord


@dataclass(frozen=True)
class NativeStyleDifficultyRecord:
    difficulty: str
    episode: NativeEpisode

    def __post_init__(self) -> None:
        if self.difficulty not in DIFFICULTIES:
            raise ValueError("Unknown native-style difficulty")

    @property
    def identity(self) -> tuple[str, str, str, int, str]:
        return (
            self.episode.policy,
            self.difficulty,
            self.episode.opponent,
            self.episode.pair_index,
            self.episode.learner_side,
        )

    def to_dict(self) -> dict[str, object]:
        return {"difficulty": self.difficulty, **self.episode.to_dict()}

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        style: NativeStyle,
    ) -> NativeStyleDifficultyRecord:
        record_type = _record_type(style)
        names = {field.name for field in fields(record_type)}
        if "difficulty" not in payload or any(name not in payload for name in names):
            raise ValueError("Native-style difficulty record is incomplete")
        episode = record_type(
            **{name: payload[name] for name in names}  # type: ignore[arg-type]
        )
        return cls(str(payload["difficulty"]), episode)


def evaluate_native_style_difficulty(
    records: Sequence[NativeStyleDifficultyRecord],
    *,
    style: NativeStyle,
    expected_pairs_per_opponent: int,
    expected_scenario_hash: str,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    if min(expected_pairs_per_opponent, bootstrap_samples) <= 0:
        raise ValueError("Native-style difficulty settings must be positive")
    record_type = _record_type(style)
    policies = _policies(style)
    if any(
        not isinstance(row.episode, record_type)
        or row.episode.policy not in policies
        for row in records
    ):
        raise ValueError("Native-style difficulty records do not match style")
    expected_per_tier = (
        len(policies) * len(DUEL_OPPONENTS) * expected_pairs_per_opponent * 2
    )
    expected = expected_per_tier * len(DIFFICULTIES)
    identities = [row.identity for row in records]
    duplicates = len(identities) - len(set(identities))

    tier_summaries: dict[str, Any] = {}
    tier_payloads: dict[str, object] = {}
    for index, difficulty in enumerate(DIFFICULTIES):
        episodes = [
            row.episode for row in records if row.difficulty == difficulty
        ]
        summary = _evaluate_tier(
            style,
            episodes,
            expected_pairs_per_opponent=expected_pairs_per_opponent,
            expected_scenario_hash=expected_scenario_hash,
            bootstrap_seed=bootstrap_seed + index,
            bootstrap_samples=bootstrap_samples,
        )
        tier_summaries[difficulty] = summary
        tier_payloads[difficulty] = asdict(summary)

    schedule_complete = (
        len(records) == expected
        and duplicates == 0
        and all(
            tier_summaries[difficulty].complete
            and tier_summaries[difficulty].episodes == expected_per_tier
            for difficulty in DIFFICULTIES
        )
    )
    cells = {
        policy: {
            difficulty: {
                "performance": tier_summaries[difficulty]
                .policies[policy]
                .performance,
                "objective_rate": tier_summaries[difficulty]
                .policies[policy]
                .objective_rate,
            }
            for difficulty in DIFFICULTIES
        }
        for policy in policies
    }
    aggregate_monotonic = all(
        _approximately_monotonic(
            [cells[policy][difficulty]["performance"] for difficulty in DIFFICULTIES]
        )
        for policy in policies
    )
    objective_capability = all(
        cells[policy]["easy"]["objective_rate"] + 1e-12
        >= 0.70 * cells[policy]["hard"]["objective_rate"]
        and cells[policy]["normal"]["objective_rate"] + 1e-12
        >= 0.85 * cells[policy]["hard"]["objective_rate"]
        for policy in policies
    )
    monotonic_opponents = {
        policy: sum(
            _approximately_monotonic(
                [
                    _opponent_performance(records, policy, difficulty, opponent)
                    for difficulty in DIFFICULTIES
                ]
            )
            for opponent in DUEL_OPPONENTS
        )
        for policy in policies
    }
    protocol_clean = duplicates == 0 and all(
        tier_summaries[difficulty].protocol_inconsistencies == 0
        for difficulty in DIFFICULTIES
    )
    gates = {
        "complete": schedule_complete,
        "protocol_clean": protocol_clean,
        "aggregate_monotonic": aggregate_monotonic,
        "per_opponent_monotonic": all(
            count >= 4 for count in monotonic_opponents.values()
        ),
        "objective_capability": objective_capability,
        "style_preserved_at_every_tier": all(
            tier_summaries[difficulty].passed for difficulty in DIFFICULTIES
        ),
    }
    return {
        "schema_version": 1,
        "stage": f"m5-{style}-difficulty",
        "style": style,
        "complete": schedule_complete,
        "passed": all(gates.values()),
        "episodes": len(records),
        "expected_episodes": expected,
        "protocol_inconsistencies": sum(
            tier_summaries[difficulty].protocol_inconsistencies
            for difficulty in DIFFICULTIES
        )
        + duplicates,
        "environment_retries": sum(
            tier_summaries[difficulty].environment_retries
            for difficulty in DIFFICULTIES
        ),
        "monotonic_opponents": monotonic_opponents,
        "gates": gates,
        "cells": cells,
        "tiers": tier_payloads,
        "test_cases_accessed": False,
    }


def _record_type(
    style: NativeStyle,
) -> type[DefensiveEpisodeRecord] | type[ExplorerEpisodeRecord]:
    if style == "defensive":
        return DefensiveEpisodeRecord
    if style == "explorer":
        return ExplorerEpisodeRecord
    raise ValueError("Unsupported native difficulty style")


def _policies(style: NativeStyle) -> tuple[str, str]:
    if style == "defensive":
        return DEFENSIVE_POLICIES
    if style == "explorer":
        return EXPLORER_POLICIES
    raise ValueError("Unsupported native difficulty style")


def _evaluate_tier(
    style: NativeStyle,
    records: Sequence[NativeEpisode],
    **kwargs: object,
) -> object:
    if style == "defensive":
        return evaluate_defensive_records(records, **kwargs)  # type: ignore[arg-type]
    return evaluate_explorer_records(records, **kwargs)  # type: ignore[arg-type]


def _approximately_monotonic(values: Sequence[float]) -> bool:
    return all(
        left <= right + MONOTONIC_TOLERANCE
        for left, right in zip(values, values[1:], strict=False)
    )


def _opponent_performance(
    records: Sequence[NativeStyleDifficultyRecord],
    policy: str,
    difficulty: str,
    opponent: str,
) -> float:
    values = [
        row.episode.performance
        for row in records
        if row.episode.policy == policy
        and row.difficulty == difficulty
        and row.episode.opponent == opponent
    ]
    if not values:
        return 0.0
    return sum(values) / len(values)
