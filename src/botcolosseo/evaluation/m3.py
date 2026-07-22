from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_protocol import DuelEventType
from botcolosseo.envs.duel_types import DuelActorObservation, DuelPrivilegedState
from botcolosseo.envs.synchronous_duel import DuelObservations, SynchronousDuelEnv
from botcolosseo.evaluation.m1 import wilson_interval
from botcolosseo.evaluation.m2 import valid_action_tic_boundary
from botcolosseo.evaluation.paired_bootstrap import (
    FROZEN_BOOTSTRAP_CONFIDENCE,
    FROZEN_BOOTSTRAP_SAMPLES,
    FROZEN_BOOTSTRAP_SEED,
    PairedBootstrapInterval,
    paired_bootstrap_difference,
)
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS
from botcolosseo.scenarios.league_splits import LeagueCase
from botcolosseo.scenarios.regions import RegionGraph

STRONG_BASE_POLICY = "strong_base"
M2_BASELINE_POLICY = "m2_baseline"
NO_OPPONENT = "no_opponent"
M3_CATEGORIES = ("script", "no_opponent", "heldout", "historical")
EXPECTED_CORE_LOCATIONS = ((0.0, 0.0), (-64.0, 64.0), (64.0, -64.0))
FROZEN_M3_THRESHOLDS = {
    "script_average_win_rate": 0.70,
    "script_per_opponent_win_rate": 0.55,
    "no_opponent_full_objective_rate": 0.90,
    "heldout_full_objective_rate": 0.80,
    "paired_score_lcb": 0.0,
}


class NoOpponentController:
    name = NO_OPPONENT

    def reset(self, *, seed: int) -> None:
        del seed

    def act(self, observation: object, state: object) -> MacroAction:
        del observation, state
        return MacroAction.IDLE


class M3Controller(Protocol):
    def reset(self, *, seed: int) -> None: ...

    def act(
        self,
        observation: DuelActorObservation,
        privileged_state: Callable[[], DuelPrivilegedState],
    ) -> MacroAction: ...


class NoOpponentDuelController:
    def reset(self, *, seed: int) -> None:
        del seed

    def act(
        self,
        observation: DuelActorObservation,
        privileged_state: Callable[[], DuelPrivilegedState],
    ) -> MacroAction:
        del observation, privileged_state
        return MacroAction.IDLE


@dataclass(frozen=True)
class M3EpisodeRecord:
    policy: str
    category: str
    split: str
    opponent: str
    pair_index: int
    seed: int
    learner_side: str
    outcome: str
    objective_completed: bool
    goal_reached: bool
    pickup_completed: bool
    return_completed: bool
    valid_hit: bool
    disengage_success: bool
    learner_score: int
    opponent_score: int
    actual_core_x: float
    actual_core_y: float
    decisions: int
    terminated: bool
    truncated: bool
    peer_tic_lag_max: int
    protocol_inconsistent: bool
    action_tic_inconsistent: bool
    score_event_inconsistent: bool
    fairness_schema_inconsistent: bool
    scenario_hash: str
    environment_attempts: int

    @property
    def score_difference(self) -> int:
        return self.learner_score - self.opponent_score

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "score_difference": self.score_difference}


def run_m3_episode(
    case: LeagueCase,
    *,
    policy_name: str,
    category: str,
    opponent_name: str,
    learner: M3Controller,
    opponent: M3Controller,
    graph: RegionGraph,
    config_path: Path,
    max_decisions: int,
    environment_factory: Callable[[LeagueCase], SynchronousDuelEnv] | None = None,
) -> M3EpisodeRecord:
    if category not in M3_CATEGORIES or max_decisions <= 0:
        raise ValueError("Invalid M3 episode category or decision limit")
    environment = (
        environment_factory(case)
        if environment_factory is not None
        else SynchronousDuelEnv(
            config_path=config_path,
            region_graph=graph,
            seed=case.seed,
            max_decisions=max_decisions,
        )
    )
    action_tic_inconsistent = False
    score_event_counts: Counter[str] = Counter()
    learner_events: Counter[DuelEventType] = Counter()
    peer_tic_lag_max = 0
    decisions = 0
    terminated = False
    truncated = False
    goal_reached = False
    low_health_seen = False
    disengage_success = False
    try:
        observations, reset_info = environment.reset()
        environment.set_shaping_scale(0.0)
        learner.reset(seed=case.seed ^ 0xA5A5A5A5)
        opponent.reset(seed=case.seed ^ 0x5A5A5A5A)
        initial_state = environment.teacher_state()
        actual_core = (initial_state.core_x, initial_state.core_y)
        initial_scores = {
            "host": observations.host.own_score,
            "opponent": observations.opponent.own_score,
        }
        while not (terminated or truncated):
            state = environment.teacher_state()
            learner_observation = (
                observations.host
                if case.learner_side == "host"
                else observations.opponent
            )
            opponent_observation = (
                observations.opponent
                if case.learner_side == "host"
                else observations.host
            )
            learner_action = learner.act(learner_observation, environment.teacher_state)
            opponent_action = opponent.act(
                opponent_observation, environment.teacher_state
            )
            host_action, away_action = (
                (learner_action, opponent_action)
                if case.learner_side == "host"
                else (opponent_action, learner_action)
            )
            step = environment.step(host_action, away_action)
            observations = DuelObservations(step.host, step.opponent)
            decisions += 1
            terminated, truncated = step.terminated, step.truncated
            peer_tic_lag_max = max(peer_tic_lag_max, step.peer_tic_lag)
            action_tic_inconsistent |= not valid_action_tic_boundary(
                step.action_tics,
                terminated=terminated,
                truncated=truncated,
            )
            for event in step.events:
                if event.type is DuelEventType.SCORE:
                    score_event_counts[event.side] += 1
                if event.side == case.learner_side:
                    learner_events[event.type] += 1
            own_x, own_y, own_health = (
                (state.host_x, state.host_y, state.host_health)
                if case.learner_side == "host"
                else (state.opponent_x, state.opponent_y, state.opponent_health)
            )
            other_x, other_y = (
                (state.opponent_x, state.opponent_y)
                if case.learner_side == "host"
                else (state.host_x, state.host_y)
            )
            goal_reached |= math.hypot(state.core_x - own_x, state.core_y - own_y) <= 64.0
            low_health_seen |= own_health <= 25.0
            disengage_success |= low_health_seen and math.hypot(
                other_x - own_x, other_y - own_y
            ) >= 256.0
        learner_observation = (
            observations.host
            if case.learner_side == "host"
            else observations.opponent
        )
        learner_score = learner_observation.own_score
        opponent_score = learner_observation.opponent_score
        final_scores = {
            "host": observations.host.own_score,
            "opponent": observations.opponent.own_score,
        }
        score_event_inconsistent = any(
            score_event_counts[side] != final_scores[side] - initial_scores[side]
            for side in ("host", "opponent")
        )
        difference = learner_score - opponent_score
        outcome = "win" if difference > 0 else "loss" if difference < 0 else "draw"
        pickup = learner_events[DuelEventType.PICKUP] > 0
        objective = learner_events[DuelEventType.SCORE] > 0
        return M3EpisodeRecord(
            policy=policy_name,
            category=category,
            split=case.split,
            opponent=opponent_name,
            pair_index=case.pair_index,
            seed=case.seed,
            learner_side=case.learner_side,
            outcome=outcome,
            objective_completed=objective,
            goal_reached=goal_reached or pickup,
            pickup_completed=pickup,
            return_completed=objective,
            valid_hit=learner_events[DuelEventType.VALID_HIT] > 0,
            disengage_success=disengage_success,
            learner_score=learner_score,
            opponent_score=opponent_score,
            actual_core_x=actual_core[0],
            actual_core_y=actual_core[1],
            decisions=decisions,
            terminated=terminated,
            truncated=truncated,
            peer_tic_lag_max=peer_tic_lag_max,
            protocol_inconsistent=False,
            action_tic_inconsistent=action_tic_inconsistent,
            score_event_inconsistent=score_event_inconsistent,
            fairness_schema_inconsistent=False,
            scenario_hash=reset_info.scenario_hash,
            environment_attempts=1,
        )
    finally:
        environment.close()


@dataclass(frozen=True)
class M3RateSummary:
    successes: int
    trials: int
    rate: float
    wilson_lower: float
    wilson_upper: float


@dataclass(frozen=True)
class M3PerformanceSummary:
    episodes: int
    wins: M3RateSummary
    objectives: M3RateSummary
    mean_score_difference: float


@dataclass(frozen=True)
class M3CategorySummary:
    episodes: int
    wins: M3RateSummary
    objectives: M3RateSummary
    mean_score_difference: float
    capabilities: dict[str, M3RateSummary]
    opponents: dict[str, M3PerformanceSummary]


@dataclass(frozen=True)
class M3EvaluationSummary:
    schema_version: int
    official: bool
    complete: bool
    passed: bool
    episodes: int
    expected_episodes: int
    pool_size: int
    protocol_inconsistencies: int
    protocol_counts: dict[str, int]
    artifact_inconsistencies: int
    environment_retries: int
    paired_historical_score_difference: PairedBootstrapInterval | None
    gates: dict[str, bool]
    categories: dict[str, M3CategorySummary]
    historical_by_policy: dict[str, dict[str, M3PerformanceSummary]]
    heldout_core_strata: dict[str, M3PerformanceSummary]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def expected_m3_episode_count(pool_size: int) -> int:
    if not 8 <= pool_size <= 12:
        raise ValueError("Official M3 pool size must be between 8 and 12")
    return 500 + 100 + 100 + pool_size * 20 * 2 * 2


def _rate(successes: int, trials: int) -> M3RateSummary:
    lower, upper = wilson_interval(successes, trials)
    return M3RateSummary(successes, trials, successes / trials, lower, upper)


def _performance(records: Sequence[M3EpisodeRecord]) -> M3PerformanceSummary:
    trials = len(records)
    return M3PerformanceSummary(
        episodes=trials,
        wins=_rate(sum(record.outcome == "win" for record in records), trials),
        objectives=_rate(sum(record.objective_completed for record in records), trials),
        mean_score_difference=float(
            np.mean([record.score_difference for record in records])
        ),
    )


def _category(records: Sequence[M3EpisodeRecord]) -> M3CategorySummary:
    base = _performance(records)
    fields = {
        "goal_reach": "goal_reached",
        "pickup": "pickup_completed",
        "return": "return_completed",
        "valid_hit": "valid_hit",
        "disengage": "disengage_success",
        "full_objective": "objective_completed",
    }
    capabilities = {
        name: _rate(
            sum(bool(getattr(record, field)) for record in records), len(records)
        )
        for name, field in fields.items()
    }
    opponents = {
        opponent: _performance(
            [record for record in records if record.opponent == opponent]
        )
        for opponent in sorted({record.opponent for record in records})
    }
    return M3CategorySummary(
        episodes=base.episodes,
        wins=base.wins,
        objectives=base.objectives,
        mean_score_difference=base.mean_score_difference,
        capabilities=capabilities,
        opponents=opponents,
    )


def _paired(records: Sequence[M3EpisodeRecord], *, expected_pairs: int) -> bool:
    grouped: dict[int, list[M3EpisodeRecord]] = defaultdict(list)
    for record in records:
        grouped[record.pair_index].append(record)
    return len(grouped) == expected_pairs and all(
        len(pair) == 2
        and {record.learner_side for record in pair} == {"host", "opponent"}
        and len({record.seed for record in pair}) == 1
        for pair in grouped.values()
    )


def _schedule_complete(
    records: Sequence[M3EpisodeRecord], historical_policy_ids: tuple[str, ...]
) -> bool:
    scripts = [record for record in records if record.category == "script"]
    no_opponent = [
        record for record in records if record.category == "no_opponent"
    ]
    heldout = [record for record in records if record.category == "heldout"]
    historical = [record for record in records if record.category == "historical"]
    if not all(
        _paired(
            [record for record in scripts if record.opponent == opponent],
            expected_pairs=50,
        )
        for opponent in DUEL_OPPONENTS
    ):
        return False
    if not _paired(no_opponent, expected_pairs=50):
        return False
    if not all(
        _paired(
            [record for record in heldout if record.opponent == opponent],
            expected_pairs=10,
        )
        for opponent in DUEL_OPPONENTS
    ):
        return False
    historical_keys: list[set[tuple[str, int, str]]] = []
    historical_seeds: list[dict[tuple[str, int, str], int]] = []
    for policy in (STRONG_BASE_POLICY, M2_BASELINE_POLICY):
        policy_records = [record for record in historical if record.policy == policy]
        if not all(
            _paired(
                [record for record in policy_records if record.opponent == opponent],
                expected_pairs=20,
            )
            for opponent in historical_policy_ids
        ):
            return False
        historical_keys.append(
            {
                (record.opponent, record.pair_index, record.learner_side)
                for record in policy_records
            }
        )
        historical_seeds.append(
            {
                (record.opponent, record.pair_index, record.learner_side): record.seed
                for record in policy_records
            }
        )
    return (
        historical_keys[0] == historical_keys[1]
        and historical_seeds[0] == historical_seeds[1]
    )


def _historical_paired_scores(
    records: Sequence[M3EpisodeRecord], historical_policy_ids: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray] | None:
    grouped: dict[tuple[str, str, int], list[M3EpisodeRecord]] = defaultdict(list)
    for record in records:
        if record.category == "historical":
            grouped[(record.policy, record.opponent, record.pair_index)].append(record)
    strong: list[float] = []
    baseline: list[float] = []
    for opponent in historical_policy_ids:
        pair_indices = sorted(
            {
                pair_index
                for policy, selected_opponent, pair_index in grouped
                if policy == STRONG_BASE_POLICY and selected_opponent == opponent
            }
        )
        if len(pair_indices) != 20:
            return None
        for pair_index in pair_indices:
            strong_rows = grouped.get((STRONG_BASE_POLICY, opponent, pair_index), [])
            baseline_rows = grouped.get((M2_BASELINE_POLICY, opponent, pair_index), [])
            if len(strong_rows) != 2 or len(baseline_rows) != 2:
                return None
            strong.append(float(np.mean([row.score_difference for row in strong_rows])))
            baseline.append(
                float(np.mean([row.score_difference for row in baseline_rows]))
            )
    if not strong:
        return None
    return np.asarray(strong), np.asarray(baseline)


def _at_least(value: float, threshold: float) -> bool:
    return value + 1e-12 >= threshold


def _core_key(x: float, y: float) -> str:
    return f"{x:g},{y:g}"


def evaluate_m3_records(
    records: Sequence[M3EpisodeRecord],
    *,
    historical_policy_ids: Sequence[str],
    expected_scenario_hash: str,
    official: bool = True,
    artifact_inconsistencies: int = 0,
    bootstrap_seed: int = FROZEN_BOOTSTRAP_SEED,
    bootstrap_samples: int = FROZEN_BOOTSTRAP_SAMPLES,
    bootstrap_confidence: float = FROZEN_BOOTSTRAP_CONFIDENCE,
) -> M3EvaluationSummary:
    historical_ids = tuple(sorted(historical_policy_ids))
    if (
        len(set(historical_ids)) != len(historical_ids)
        or not 8 <= len(historical_ids) <= 12
    ):
        raise ValueError("Official M3 requires 8 to 12 unique historical policies")
    if not expected_scenario_hash or artifact_inconsistencies < 0:
        raise ValueError("Invalid M3 evaluation provenance")
    if official and (
        bootstrap_seed != FROZEN_BOOTSTRAP_SEED
        or bootstrap_samples != FROZEN_BOOTSTRAP_SAMPLES
        or bootstrap_confidence != FROZEN_BOOTSTRAP_CONFIDENCE
    ):
        raise ValueError("Official M3 requires the frozen paired bootstrap")
    expected_episodes = expected_m3_episode_count(len(historical_ids))
    identities = [
        (
            record.policy,
            record.category,
            record.opponent,
            record.pair_index,
            record.learner_side,
        )
        for record in records
    ]
    protocol_counts: Counter[str] = Counter()
    protocol_counts["duplicate_rows"] = len(identities) - len(set(identities))
    for record in records:
        if record.environment_attempts <= 0:
            raise ValueError("environment_attempts must be positive")
        score_sign = (record.score_difference > 0) - (record.score_difference < 0)
        expected_outcome = {1: "win", 0: "draw", -1: "loss"}[score_sign]
        allowed_policy = record.policy == STRONG_BASE_POLICY or (
            record.category == "historical" and record.policy == M2_BASELINE_POLICY
        )
        allowed_opponent = (
            record.opponent in DUEL_OPPONENTS
            if record.category in ("script", "heldout")
            else record.opponent == NO_OPPONENT
            if record.category == "no_opponent"
            else record.opponent in historical_ids
        )
        expected_split = "heldout" if record.category == "heldout" else "test"
        protocol_counts["explicit_inconsistency_rows"] += int(
            record.protocol_inconsistent
        )
        protocol_counts["action_tic_inconsistency_rows"] += int(
            record.action_tic_inconsistent
        )
        protocol_counts["score_event_inconsistency_rows"] += int(
            record.score_event_inconsistent
        )
        protocol_counts["fairness_schema_rows"] += int(
            record.fairness_schema_inconsistent
            or record.category not in M3_CATEGORIES
            or record.learner_side not in ("host", "opponent")
            or not allowed_policy
            or not allowed_opponent
            or record.split != expected_split
        )
        protocol_counts["peer_tic_lag_rows"] += int(record.peer_tic_lag_max != 0)
        protocol_counts["scenario_mismatch_rows"] += int(
            record.scenario_hash != expected_scenario_hash
        )
        protocol_counts["outcome_score_mismatch_rows"] += int(
            record.outcome != expected_outcome
        )
        protocol_counts["invalid_terminal_boundary_rows"] += int(
            record.terminated == record.truncated
        )
        protocol_counts["nonfinite_core_rows"] += int(
            not math.isfinite(record.actual_core_x)
            or not math.isfinite(record.actual_core_y)
        )
        protocol_counts["environment_retry_rows"] += int(
            record.environment_attempts > 1
        )
    issue_names = (
        "duplicate_rows",
        "explicit_inconsistency_rows",
        "action_tic_inconsistency_rows",
        "score_event_inconsistency_rows",
        "fairness_schema_rows",
        "peer_tic_lag_rows",
        "scenario_mismatch_rows",
        "outcome_score_mismatch_rows",
        "invalid_terminal_boundary_rows",
        "nonfinite_core_rows",
    )
    protocol_inconsistencies = sum(protocol_counts[name] for name in issue_names)
    by_category = {
        category: [record for record in records if record.category == category]
        for category in M3_CATEGORIES
    }
    categories = {
        category: _category(category_rows)
        for category, category_rows in by_category.items()
        if category_rows
    }
    historical_by_policy = {
        opponent: {
            policy: _performance(
                [
                    record
                    for record in by_category["historical"]
                    if record.opponent == opponent and record.policy == policy
                ]
            )
            for policy in (STRONG_BASE_POLICY, M2_BASELINE_POLICY)
            if any(
                record.opponent == opponent and record.policy == policy
                for record in by_category["historical"]
            )
        }
        for opponent in historical_ids
    }
    heldout_core_strata = {
        _core_key(x, y): _performance(
            [
                record
                for record in by_category["heldout"]
                if (record.actual_core_x, record.actual_core_y) == (x, y)
            ]
        )
        for x, y in sorted(
            {
                (record.actual_core_x, record.actual_core_y)
                for record in by_category["heldout"]
            }
        )
    }
    expected_counts = {
        "script": 500,
        "no_opponent": 100,
        "heldout": 100,
        "historical": len(historical_ids) * 80,
    }
    counts_complete = len(records) == expected_episodes and all(
        len(by_category[category]) == count
        for category, count in expected_counts.items()
    )
    schedule_complete = counts_complete and _schedule_complete(records, historical_ids)
    paired_scores = _historical_paired_scores(records, historical_ids)
    paired_interval = (
        paired_bootstrap_difference(
            paired_scores[0],
            paired_scores[1],
            seed=bootstrap_seed,
            samples=bootstrap_samples,
            confidence=bootstrap_confidence,
        )
        if paired_scores is not None
        else None
    )
    performance_available = set(categories) == set(M3_CATEGORIES) and all(
        set(historical_by_policy[opponent])
        == {STRONG_BASE_POLICY, M2_BASELINE_POLICY}
        for opponent in historical_ids
    )
    script_average = performance_available and _at_least(
        categories["script"].wins.rate,
        FROZEN_M3_THRESHOLDS["script_average_win_rate"],
    )
    script_floor = performance_available and all(
        _at_least(
            categories["script"].opponents[opponent].wins.rate,
            FROZEN_M3_THRESHOLDS["script_per_opponent_win_rate"],
        )
        for opponent in DUEL_OPPONENTS
        if opponent in categories["script"].opponents
    ) and all(opponent in categories["script"].opponents for opponent in DUEL_OPPONENTS)
    no_opponent_gate = performance_available and _at_least(
        categories["no_opponent"].objectives.rate,
        FROZEN_M3_THRESHOLDS["no_opponent_full_objective_rate"],
    )
    heldout_gate = performance_available and _at_least(
        categories["heldout"].objectives.rate,
        FROZEN_M3_THRESHOLDS["heldout_full_objective_rate"],
    )
    historical_improved = performance_available and min(
        historical_by_policy[opponent][STRONG_BASE_POLICY].wins.rate
        for opponent in historical_ids
    ) > min(
        historical_by_policy[opponent][M2_BASELINE_POLICY].wins.rate
        for opponent in historical_ids
    )
    paired_gate = (
        paired_interval is not None
        and _at_least(
            paired_interval.lower, FROZEN_M3_THRESHOLDS["paired_score_lcb"]
        )
    )
    actual_cores = {
        (record.actual_core_x, record.actual_core_y)
        for record in by_category["heldout"]
    }
    core_strata_complete = actual_cores == set(EXPECTED_CORE_LOCATIONS)
    confidence_intervals_finite = paired_interval is not None and all(
        math.isfinite(value)
        for value in (
            paired_interval.estimate,
            paired_interval.lower,
            paired_interval.upper,
        )
    )
    complete = schedule_complete and core_strata_complete
    gates = {
        "official": official,
        "complete": complete,
        "pool_size": 8 <= len(historical_ids) <= 12,
        "protocol_clean": protocol_inconsistencies == 0,
        "artifact_clean": artifact_inconsistencies == 0,
        "script_average_win_rate": bool(script_average),
        "script_per_opponent_win_rate": bool(script_floor),
        "no_opponent_full_objective_rate": bool(no_opponent_gate),
        "heldout_full_objective_rate": bool(heldout_gate),
        "historical_worst_case_improved": bool(historical_improved),
        "paired_score_lcb": bool(paired_gate),
        "heldout_core_strata_complete": core_strata_complete,
        "confidence_intervals_finite": confidence_intervals_finite,
    }
    return M3EvaluationSummary(
        schema_version=1,
        official=official,
        complete=complete,
        passed=all(gates.values()),
        episodes=len(records),
        expected_episodes=expected_episodes,
        pool_size=len(historical_ids),
        protocol_inconsistencies=protocol_inconsistencies,
        protocol_counts=dict(sorted(protocol_counts.items())),
        artifact_inconsistencies=artifact_inconsistencies,
        environment_retries=sum(record.environment_attempts - 1 for record in records),
        paired_historical_score_difference=paired_interval,
        gates=gates,
        categories=categories,
        historical_by_policy=historical_by_policy,
        heldout_core_strata=heldout_core_strata,
    )
