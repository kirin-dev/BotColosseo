from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from botcolosseo.envs.duel_protocol import DuelEventType
from botcolosseo.envs.synchronous_duel import SynchronousDuelEnv
from botcolosseo.evaluation.m2 import (
    EvaluationPolicy,
    TeacherEvaluationPolicy,
    paired_bootstrap_interval,
    valid_action_tic_boundary,
)
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS, DuelCase
from botcolosseo.scenarios.regions import RegionGraph

EXPLORER_POLICIES = ("strong_base", "explorer")
EXPLORER_ROUTES = ("upper", "lower", "flank")
ROUTE_ENTROPY_ESTIMATOR = "paired_episode_normalized_entropy_v1"


def classify_completed_route(regions: Sequence[str]) -> str:
    visited = set(regions)
    if {"flank_west", "flank_east"}.issubset(visited):
        return "flank"
    if "upper_route" in visited:
        return "upper"
    if "lower_route" in visited:
        return "lower"
    return "mixed_or_unknown"


def normalized_route_entropy(counts: Sequence[int]) -> float:
    values = np.asarray(counts, dtype=np.float64)
    if values.shape != (len(EXPLORER_ROUTES),) or np.any(values < 0):
        raise ValueError("Explorer route entropy counts are invalid")
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probabilities = values[values > 0] / total
    return float(-(probabilities * np.log(probabilities)).sum() / math.log(3.0))


@dataclass(frozen=True)
class ExplorerEpisodeRecord:
    policy: str
    split: str
    opponent: str
    pair_index: int
    seed: int
    learner_side: str
    outcome: str
    objective_completed: bool
    learner_score: int
    opponent_score: int
    learner_scores: int
    decisions: int
    upper_completions: int
    lower_completions: int
    flank_completions: int
    mixed_or_unknown_completions: int
    unique_regions: int
    terminated: bool
    truncated: bool
    peer_tic_lag_max: int
    protocol_inconsistent: bool
    action_tic_inconsistent: bool
    score_event_inconsistent: bool
    scenario_hash: str
    environment_attempts: int = 1

    def __post_init__(self) -> None:
        counters = (
            self.pair_index,
            self.seed,
            self.learner_score,
            self.opponent_score,
            self.learner_scores,
            self.decisions,
            self.upper_completions,
            self.lower_completions,
            self.flank_completions,
            self.mixed_or_unknown_completions,
            self.unique_regions,
            self.peer_tic_lag_max,
        )
        if self.policy not in EXPLORER_POLICIES:
            raise ValueError("Unknown Explorer evaluation policy")
        if self.split != "validation" or self.opponent not in DUEL_OPPONENTS:
            raise ValueError("Explorer evaluation requires frozen validation scripts")
        if self.learner_side not in ("host", "opponent"):
            raise ValueError("Explorer evaluation learner side is invalid")
        if any(value < 0 for value in counters) or self.environment_attempts <= 0:
            raise ValueError("Explorer evaluation counters must be nonnegative")
        if self.decisions <= 0 or self.learner_scores > self.learner_score:
            raise ValueError("Explorer score accounting is invalid")
        if self.completed_routes + self.mixed_or_unknown_completions != self.learner_scores:
            raise ValueError("Explorer route completion accounting is invalid")
        expected = (
            "win"
            if self.score_difference > 0
            else "loss"
            if self.score_difference < 0
            else "draw"
        )
        if self.outcome != expected:
            raise ValueError("Explorer evaluation outcome does not match score")
        if self.terminated == self.truncated or not self.scenario_hash:
            raise ValueError("Explorer terminal or scenario metadata is invalid")

    @property
    def score_difference(self) -> int:
        return self.learner_score - self.opponent_score

    @property
    def performance(self) -> float:
        points = {"win": 1.0, "draw": 0.5, "loss": 0.0}[self.outcome]
        return 0.5 * (points + float(self.objective_completed))

    @property
    def completed_routes(self) -> int:
        return self.upper_completions + self.lower_completions + self.flank_completions

    @property
    def route_entropy(self) -> float:
        return normalized_route_entropy(
            (self.upper_completions, self.lower_completions, self.flank_completions)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "score_difference": self.score_difference,
            "performance": self.performance,
            "completed_routes": self.completed_routes,
            "route_entropy": self.route_entropy,
        }


@dataclass(frozen=True)
class ExplorerPolicySummary:
    episodes: int
    win_rate: float
    objective_rate: float
    mean_score_difference: float
    performance: float
    learner_scores: int
    decisions_per_score: float | None
    route_entropy: float
    mean_episode_route_entropy: float
    upper_completions: int
    lower_completions: int
    flank_completions: int
    mixed_or_unknown_completions: int
    flank_completion_rate: float
    mean_unique_regions: float


@dataclass(frozen=True)
class ExplorerEvaluationSummary:
    complete: bool
    passed: bool
    episodes: int
    expected_episodes: int
    protocol_inconsistencies: int
    environment_retries: int
    skill_retention: float
    per_opponent_retention: dict[str, float]
    route_entropy_estimator: str
    route_entropy_delta: float
    route_entropy_delta_ci: tuple[float, float] | None
    gates: dict[str, bool]
    policies: dict[str, ExplorerPolicySummary]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _retention(style: float, base: float) -> float:
    return style / base if base > 1e-12 else float(style + 1e-12 >= base)


def _policy_summary(records: Sequence[ExplorerEpisodeRecord]) -> ExplorerPolicySummary:
    episodes = len(records)
    scores = sum(row.learner_scores for row in records)
    decisions = sum(row.decisions for row in records)
    upper = sum(row.upper_completions for row in records)
    lower = sum(row.lower_completions for row in records)
    flank = sum(row.flank_completions for row in records)
    mixed = sum(row.mixed_or_unknown_completions for row in records)
    return ExplorerPolicySummary(
        episodes=episodes,
        win_rate=sum(row.outcome == "win" for row in records) / episodes,
        objective_rate=sum(row.objective_completed for row in records) / episodes,
        mean_score_difference=float(np.mean([row.score_difference for row in records])),
        performance=float(np.mean([row.performance for row in records])),
        learner_scores=scores,
        decisions_per_score=decisions / scores if scores else None,
        route_entropy=normalized_route_entropy((upper, lower, flank)),
        mean_episode_route_entropy=float(np.mean([row.route_entropy for row in records])),
        upper_completions=upper,
        lower_completions=lower,
        flank_completions=flank,
        mixed_or_unknown_completions=mixed,
        flank_completion_rate=flank / scores if scores else 0.0,
        mean_unique_regions=float(np.mean([row.unique_regions for row in records])),
    )


def _schedule_complete(
    records: Sequence[ExplorerEpisodeRecord], *, pairs_per_opponent: int
) -> bool:
    expected = len(DUEL_OPPONENTS) * pairs_per_opponent * 2
    for policy in EXPLORER_POLICIES:
        selected = [row for row in records if row.policy == policy]
        keys = {(row.opponent, row.pair_index, row.learner_side) for row in selected}
        if len(selected) != expected or len(keys) != expected:
            return False
        if Counter(row.opponent for row in selected) != Counter(
            {opponent: pairs_per_opponent * 2 for opponent in DUEL_OPPONENTS}
        ):
            return False
        sides: dict[tuple[str, int], set[str]] = defaultdict(set)
        for row in selected:
            sides[(row.opponent, row.pair_index)].add(row.learner_side)
        if any(value != {"host", "opponent"} for value in sides.values()):
            return False
    return True


def evaluate_explorer_records(
    records: Sequence[ExplorerEpisodeRecord],
    *,
    expected_pairs_per_opponent: int,
    expected_scenario_hash: str,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> ExplorerEvaluationSummary:
    if expected_pairs_per_opponent <= 0:
        raise ValueError("Expected Explorer evaluation pairs must be positive")
    expected = len(EXPLORER_POLICIES) * len(DUEL_OPPONENTS) * expected_pairs_per_opponent * 2
    identities = [
        (row.policy, row.opponent, row.pair_index, row.learner_side)
        for row in records
    ]
    inconsistencies = len(identities) - len(set(identities))
    inconsistencies += sum(
        int(
            row.protocol_inconsistent
            or row.action_tic_inconsistent
            or row.score_event_inconsistent
            or row.peer_tic_lag_max != 0
            or row.scenario_hash != expected_scenario_hash
        )
        for row in records
    )
    complete = len(records) == expected and _schedule_complete(
        records, pairs_per_opponent=expected_pairs_per_opponent
    )
    by_policy = {
        policy: [row for row in records if row.policy == policy]
        for policy in EXPLORER_POLICIES
    }
    policies = {
        policy: _policy_summary(rows) for policy, rows in by_policy.items() if rows
    }
    available = set(policies) == set(EXPLORER_POLICIES)
    retention = (
        _retention(policies["explorer"].performance, policies["strong_base"].performance)
        if available
        else 0.0
    )
    per_opponent = {
        opponent: _retention(
            float(
                np.mean(
                    [
                        row.performance
                        for row in by_policy["explorer"]
                        if row.opponent == opponent
                    ]
                )
            ),
            float(
                np.mean(
                    [
                        row.performance
                        for row in by_policy["strong_base"]
                        if row.opponent == opponent
                    ]
                )
            ),
        )
        for opponent in DUEL_OPPONENTS
        if all(any(row.opponent == opponent for row in by_policy[p]) for p in EXPLORER_POLICIES)
    }
    grouped: dict[tuple[str, int, str], dict[str, ExplorerEpisodeRecord]] = defaultdict(dict)
    for row in records:
        grouped[(row.opponent, row.pair_index, row.learner_side)][row.policy] = row
    differences = None
    if grouped and all(set(pair) == set(EXPLORER_POLICIES) for pair in grouped.values()):
        differences = np.asarray(
            [
                pair["explorer"].route_entropy - pair["strong_base"].route_entropy
                for _, pair in sorted(grouped.items())
            ],
            dtype=np.float64,
        )
    interval = (
        paired_bootstrap_interval(
            differences, seed=bootstrap_seed, samples=bootstrap_samples
        )
        if differences is not None
        else None
    )
    delta = float(np.mean(differences)) if differences is not None else 0.0
    base = policies.get("strong_base")
    explorer = policies.get("explorer")
    efficiency_valid = (
        base is not None
        and explorer is not None
        and base.decisions_per_score is not None
        and explorer.decisions_per_score is not None
    )
    gates = {
        "complete": complete,
        "protocol_clean": inconsistencies == 0,
        "skill_retention": retention + 1e-12 >= 0.85,
        "per_opponent_retention": len(per_opponent) == len(DUEL_OPPONENTS)
        and all(value + 1e-12 >= 0.75 for value in per_opponent.values()),
        "route_entropy_shift": interval is not None and interval[0] > 0,
        "route_coverage": explorer is not None
        and all(
            value > 0
            for value in (
                explorer.upper_completions,
                explorer.lower_completions,
                explorer.flank_completions,
            )
        ),
        "flank_improved": base is not None
        and explorer is not None
        and explorer.flank_completion_rate > base.flank_completion_rate,
        "objective_retention": base is not None
        and explorer is not None
        and explorer.objective_rate + 1e-12 >= 0.85 * base.objective_rate,
        "efficiency_controlled": efficiency_valid
        and explorer.decisions_per_score <= 1.35 * base.decisions_per_score,
    }
    return ExplorerEvaluationSummary(
        complete=complete,
        passed=all(gates.values()),
        episodes=len(records),
        expected_episodes=expected,
        protocol_inconsistencies=inconsistencies,
        environment_retries=sum(row.environment_attempts - 1 for row in records),
        skill_retention=retention,
        per_opponent_retention=per_opponent,
        route_entropy_estimator=ROUTE_ENTROPY_ESTIMATOR,
        route_entropy_delta=delta,
        route_entropy_delta_ci=interval,
        gates=gates,
        policies=policies,
    )


EnvironmentFactory = Callable[[DuelCase], SynchronousDuelEnv]


def run_explorer_episode(
    case: DuelCase,
    *,
    policy: EvaluationPolicy,
    graph: RegionGraph,
    config_path: Path,
    max_decisions: int,
    environment_factory: EnvironmentFactory | None = None,
) -> ExplorerEpisodeRecord:
    if case.split != "validation" or max_decisions <= 0:
        raise ValueError("Explorer episode requires validation and a positive limit")
    env = (
        environment_factory(case)
        if environment_factory
        else SynchronousDuelEnv(
            config_path=config_path,
            region_graph=graph,
            seed=case.seed,
            max_decisions=max_decisions,
        )
    )
    opponent_side = "opponent" if case.learner_side == "host" else "host"
    opponent = TeacherEvaluationPolicy(case.opponent, graph, side=opponent_side)
    route_counts: Counter[str] = Counter()
    score_events: Counter[str] = Counter()
    visited_regions: set[str] = set()
    carried_regions: list[str] = []
    action_tic_inconsistent = False
    peer_tic_lag_max = 0
    terminated = truncated = False
    decisions = 0
    try:
        observations, reset_info = env.reset()
        env.set_shaping_scale(0.0)
        policy.reset(seed=case.seed ^ 0xA5A5A5A5)
        opponent.reset(seed=case.seed ^ 0x5A5A5A5A)
        learner_observation = (
            observations.host
            if case.learner_side == "host"
            else observations.opponent
        )
        initial_score = learner_observation.own_score
        initial_scores = {
            "host": observations.host.own_score,
            "opponent": observations.opponent.own_score,
        }
        learner_carrier = 1 if case.learner_side == "host" else 2
        while not (terminated or truncated):
            state = env.teacher_state()
            learner_region = (
                state.host_region
                if case.learner_side == "host"
                else state.opponent_region
            )
            if learner_region is not None:
                visited_regions.add(learner_region)
                if state.carrier == learner_carrier:
                    carried_regions.append(learner_region)
            decisions += 1
            learner_observation = (
                observations.host
                if case.learner_side == "host"
                else observations.opponent
            )
            learner_action = policy.act(learner_observation, state)
            other_observation = (
                observations.opponent
                if case.learner_side == "host"
                else observations.host
            )
            other_action = opponent.act(other_observation, state)
            host_action, away_action = (
                (learner_action, other_action)
                if case.learner_side == "host"
                else (other_action, learner_action)
            )
            step = env.step(host_action, away_action)
            observations = type(observations)(step.host, step.opponent)
            terminated, truncated = step.terminated, step.truncated
            peer_tic_lag_max = max(peer_tic_lag_max, step.peer_tic_lag)
            score_events.update(
                event.side for event in step.events if event.type is DuelEventType.SCORE
            )
            action_tic_inconsistent |= not valid_action_tic_boundary(
                step.action_tics,
                terminated=terminated,
                truncated=truncated,
            )
            if any(
                event.side == case.learner_side
                and event.type is DuelEventType.SCORE
                for event in step.events
            ):
                route_counts[classify_completed_route(carried_regions)] += 1
                carried_regions.clear()
            elif any(
                event.side == case.learner_side
                and event.type in (DuelEventType.DROP, DuelEventType.DEATH)
                for event in step.events
            ):
                carried_regions.clear()
        learner_observation = (
            observations.host
            if case.learner_side == "host"
            else observations.opponent
        )
        final_scores = {
            "host": observations.host.own_score,
            "opponent": observations.opponent.own_score,
        }
        score_inconsistent = any(
            score_events[side] != final_scores[side] - initial_scores[side]
            for side in ("host", "opponent")
        )
        learner_score = learner_observation.own_score
        opponent_score = learner_observation.opponent_score
        learner_scores = learner_score - initial_score
        difference = learner_score - opponent_score
        return ExplorerEpisodeRecord(
            policy=policy.name,
            split=case.split,
            opponent=case.opponent,
            pair_index=case.pair_index,
            seed=case.seed,
            learner_side=case.learner_side,
            outcome="win" if difference > 0 else "loss" if difference < 0 else "draw",
            objective_completed=learner_scores > 0,
            learner_score=learner_score,
            opponent_score=opponent_score,
            learner_scores=learner_scores,
            decisions=decisions,
            upper_completions=route_counts["upper"],
            lower_completions=route_counts["lower"],
            flank_completions=route_counts["flank"],
            mixed_or_unknown_completions=route_counts["mixed_or_unknown"],
            unique_regions=len(visited_regions),
            terminated=terminated,
            truncated=truncated,
            peer_tic_lag_max=peer_tic_lag_max,
            protocol_inconsistent=False,
            action_tic_inconsistent=action_tic_inconsistent,
            score_event_inconsistent=score_inconsistent,
            scenario_hash=reset_info.scenario_hash,
        )
    finally:
        env.close()
