from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields

import numpy as np

from botcolosseo.agents.difficulty import DIFFICULTIES
from botcolosseo.evaluation.style import STYLE_POLICIES, StyleEpisodeRecord
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS

MONOTONIC_TOLERANCE = 0.03


@dataclass(frozen=True)
class DifficultyEpisodeRecord:
    difficulty: str
    episode: StyleEpisodeRecord

    def __post_init__(self) -> None:
        if self.difficulty not in DIFFICULTIES:
            raise ValueError("Unknown difficulty")

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
    def from_dict(cls, payload: dict[str, object]) -> DifficultyEpisodeRecord:
        names = {field.name for field in fields(StyleEpisodeRecord)}
        return cls(
            difficulty=str(payload["difficulty"]),
            episode=StyleEpisodeRecord(  # type: ignore[arg-type]
                **{name: payload[name] for name in names}
            ),
        )


@dataclass(frozen=True)
class DifficultyCellSummary:
    episodes: int
    win_rate: float
    objective_rate: float
    mean_score_difference: float
    performance: float
    engagement_initiations_per_100_decisions: float


@dataclass(frozen=True)
class DifficultyEvaluationSummary:
    complete: bool
    passed: bool
    episodes: int
    expected_episodes: int
    protocol_inconsistencies: int
    environment_retries: int
    monotonic_opponents: dict[str, int]
    gates: dict[str, bool]
    cells: dict[str, dict[str, DifficultyCellSummary]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _cell(records: Sequence[DifficultyEpisodeRecord]) -> DifficultyCellSummary:
    episodes = len(records)
    decisions = sum(row.episode.decisions for row in records)
    engagements = sum(row.episode.engagement_initiations for row in records)
    return DifficultyCellSummary(
        episodes=episodes,
        win_rate=sum(row.episode.outcome == "win" for row in records) / episodes,
        objective_rate=sum(row.episode.objective_completed for row in records) / episodes,
        mean_score_difference=float(
            np.mean([row.episode.score_difference for row in records])
        ),
        performance=float(np.mean([row.episode.performance for row in records])),
        engagement_initiations_per_100_decisions=100.0 * engagements / decisions,
    )


def _approximately_monotonic(values: Sequence[float]) -> bool:
    return all(
        left <= right + MONOTONIC_TOLERANCE
        for left, right in zip(values, values[1:], strict=False)
    )


def _schedule_complete(
    records: Sequence[DifficultyEpisodeRecord], *, pairs_per_opponent: int
) -> bool:
    expected_per_cell = len(DUEL_OPPONENTS) * pairs_per_opponent * 2
    for policy in STYLE_POLICIES:
        for difficulty in DIFFICULTIES:
            selected = [
                row
                for row in records
                if row.episode.policy == policy and row.difficulty == difficulty
            ]
            if len(selected) != expected_per_cell:
                return False
            keys = {
                (
                    row.episode.opponent,
                    row.episode.pair_index,
                    row.episode.learner_side,
                )
                for row in selected
            }
            if len(keys) != expected_per_cell:
                return False
            if Counter(row.episode.opponent for row in selected) != Counter(
                {
                    opponent: pairs_per_opponent * 2
                    for opponent in DUEL_OPPONENTS
                }
            ):
                return False
    return True


def evaluate_difficulty_records(
    records: Sequence[DifficultyEpisodeRecord],
    *,
    expected_pairs_per_opponent: int,
    expected_scenario_hash: str,
) -> DifficultyEvaluationSummary:
    if expected_pairs_per_opponent <= 0:
        raise ValueError("Expected difficulty evaluation pairs must be positive")
    expected = (
        len(STYLE_POLICIES)
        * len(DIFFICULTIES)
        * len(DUEL_OPPONENTS)
        * expected_pairs_per_opponent
        * 2
    )
    identities = [row.identity for row in records]
    inconsistencies = len(identities) - len(set(identities))
    inconsistencies += sum(
        int(
            row.episode.protocol_inconsistent
            or row.episode.action_tic_inconsistent
            or row.episode.score_event_inconsistent
            or row.episode.peer_tic_lag_max != 0
            or row.episode.scenario_hash != expected_scenario_hash
        )
        for row in records
    )
    complete = len(records) == expected and _schedule_complete(
        records, pairs_per_opponent=expected_pairs_per_opponent
    )
    grouped = {
        policy: {
            difficulty: [
                row
                for row in records
                if row.episode.policy == policy and row.difficulty == difficulty
            ]
            for difficulty in DIFFICULTIES
        }
        for policy in STYLE_POLICIES
    }
    available = all(
        grouped[policy][difficulty]
        for policy in STYLE_POLICIES
        for difficulty in DIFFICULTIES
    )
    cells = {
        policy: {
            difficulty: _cell(grouped[policy][difficulty])
            for difficulty in DIFFICULTIES
            if grouped[policy][difficulty]
        }
        for policy in STYLE_POLICIES
    }
    monotonic_opponents: dict[str, int] = {}
    if available:
        for policy in STYLE_POLICIES:
            count = 0
            for opponent in DUEL_OPPONENTS:
                values = [
                    float(
                        np.mean(
                            [
                                row.episode.performance
                                for row in grouped[policy][difficulty]
                                if row.episode.opponent == opponent
                            ]
                        )
                    )
                    for difficulty in DIFFICULTIES
                ]
                count += int(_approximately_monotonic(values))
            monotonic_opponents[policy] = count
    aggregate_monotonic = available and all(
        _approximately_monotonic(
            [cells[policy][difficulty].performance for difficulty in DIFFICULTIES]
        )
        for policy in STYLE_POLICIES
    )
    objective_retention = available and all(
        cells[policy]["easy"].objective_rate + 1e-12
        >= 0.70 * cells[policy]["hard"].objective_rate
        and cells[policy]["normal"].objective_rate + 1e-12
        >= 0.85 * cells[policy]["hard"].objective_rate
        for policy in STYLE_POLICIES
    )
    style_direction = available and all(
        cells["aggressive"][difficulty].engagement_initiations_per_100_decisions
        > cells["strong_base"][
            difficulty
        ].engagement_initiations_per_100_decisions
        for difficulty in DIFFICULTIES
    )
    gates = {
        "complete": complete,
        "protocol_clean": inconsistencies == 0,
        "aggregate_monotonic": aggregate_monotonic,
        "per_opponent_monotonic": len(monotonic_opponents) == len(STYLE_POLICIES)
        and all(value >= 4 for value in monotonic_opponents.values()),
        "objective_capability": objective_retention,
        "style_direction_preserved": style_direction,
    }
    return DifficultyEvaluationSummary(
        complete=complete,
        passed=all(gates.values()),
        episodes=len(records),
        expected_episodes=expected,
        protocol_inconsistencies=inconsistencies,
        environment_retries=sum(
            row.episode.environment_attempts - 1 for row in records
        ),
        monotonic_opponents=monotonic_opponents,
        gates=gates,
        cells=cells,
    )
