from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from botcolosseo.envs.actions import MacroAction
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

STYLE_POLICIES = ("strong_base", "aggressive")
RETENTION_THRESHOLD = 0.85
PER_OPPONENT_RETENTION_THRESHOLD = 0.75
VALID_ATTACK_RATE_THRESHOLD = 0.20
OBJECTIVE_CHASE_RATE_CEILING = 0.25
ENGAGEMENT_COOLDOWN = 12
_ATTACK_ACTIONS = frozenset(
    {
        MacroAction.ATTACK,
        MacroAction.FORWARD_ATTACK,
        MacroAction.TURN_LEFT_ATTACK,
        MacroAction.TURN_RIGHT_ATTACK,
    }
)


@dataclass(frozen=True)
class StyleEpisodeRecord:
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
    decisions: int
    attack_decisions: int
    valid_hits: int
    engagement_initiations: int
    forward_hits: int
    invalid_attack_decisions: int
    objective_chase_decisions: int
    retreat_decisions: int
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
            self.decisions,
            self.attack_decisions,
            self.valid_hits,
            self.engagement_initiations,
            self.forward_hits,
            self.invalid_attack_decisions,
            self.objective_chase_decisions,
            self.retreat_decisions,
            self.peer_tic_lag_max,
        )
        if self.policy not in STYLE_POLICIES:
            raise ValueError("Unknown style evaluation policy")
        if self.split != "validation" or self.opponent not in DUEL_OPPONENTS:
            raise ValueError("Style evaluation requires frozen validation scripts")
        if self.learner_side not in ("host", "opponent"):
            raise ValueError("Style evaluation learner side is invalid")
        if any(value < 0 for value in counters) or self.environment_attempts <= 0:
            raise ValueError("Style evaluation counters must be nonnegative")
        if self.decisions <= 0:
            raise ValueError("Style evaluation episode must contain decisions")
        if any(
            value > self.attack_decisions
            for value in (
                self.valid_hits,
                self.forward_hits,
                self.invalid_attack_decisions,
                self.objective_chase_decisions,
            )
        ):
            raise ValueError("Attack-derived metrics exceed attack decisions")
        difference = self.score_difference
        expected = "win" if difference > 0 else "loss" if difference < 0 else "draw"
        if self.outcome != expected:
            raise ValueError("Style evaluation outcome does not match score")
        if self.terminated == self.truncated or not self.scenario_hash:
            raise ValueError("Style evaluation terminal or scenario metadata is invalid")

    @property
    def score_difference(self) -> int:
        return self.learner_score - self.opponent_score

    @property
    def outcome_points(self) -> float:
        return {"win": 1.0, "draw": 0.5, "loss": 0.0}[self.outcome]

    @property
    def performance(self) -> float:
        return 0.5 * (self.outcome_points + float(self.objective_completed))

    @property
    def engagement_rate(self) -> float:
        return 100.0 * self.engagement_initiations / self.decisions

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "score_difference": self.score_difference,
            "performance": self.performance,
            "engagement_initiations_per_100_decisions": self.engagement_rate,
        }


@dataclass(frozen=True)
class StylePolicySummary:
    episodes: int
    win_rate: float
    objective_rate: float
    mean_score_difference: float
    performance: float
    attack_rate: float
    valid_attack_rate: float
    engagement_initiations_per_100_decisions: float
    forward_hit_rate: float
    invalid_attack_rate: float
    objective_chase_rate: float
    retreat_rate: float


@dataclass(frozen=True)
class AggressiveEvaluationSummary:
    complete: bool
    passed: bool
    episodes: int
    expected_episodes: int
    protocol_inconsistencies: int
    environment_retries: int
    skill_retention: float
    per_opponent_retention: dict[str, float]
    engagement_initiation_delta: float
    engagement_initiation_delta_ci: tuple[float, float] | None
    gates: dict[str, bool]
    policies: dict[str, StylePolicySummary]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-12:
        return 1.0 if numerator + 1e-12 >= denominator else 0.0
    return numerator / denominator


def _event_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _policy_summary(records: Sequence[StyleEpisodeRecord]) -> StylePolicySummary:
    episodes = len(records)
    decisions = sum(record.decisions for record in records)
    attacks = sum(record.attack_decisions for record in records)
    valid_hits = sum(record.valid_hits for record in records)
    engagements = sum(record.engagement_initiations for record in records)
    forward_hits = sum(record.forward_hits for record in records)
    invalid_attacks = sum(record.invalid_attack_decisions for record in records)
    objective_chases = sum(record.objective_chase_decisions for record in records)
    retreats = sum(record.retreat_decisions for record in records)
    return StylePolicySummary(
        episodes=episodes,
        win_rate=sum(record.outcome == "win" for record in records) / episodes,
        objective_rate=sum(record.objective_completed for record in records) / episodes,
        mean_score_difference=float(np.mean([record.score_difference for record in records])),
        performance=float(np.mean([record.performance for record in records])),
        attack_rate=attacks / decisions,
        valid_attack_rate=_event_rate(valid_hits, attacks),
        engagement_initiations_per_100_decisions=100.0 * engagements / decisions,
        forward_hit_rate=_event_rate(forward_hits, attacks),
        invalid_attack_rate=_event_rate(invalid_attacks, attacks),
        objective_chase_rate=_event_rate(objective_chases, attacks),
        retreat_rate=retreats / decisions,
    )


def _paired_engagement_differences(
    records: Sequence[StyleEpisodeRecord],
) -> np.ndarray | None:
    grouped: dict[tuple[str, int, str], dict[str, StyleEpisodeRecord]] = defaultdict(dict)
    for record in records:
        key = (record.opponent, record.pair_index, record.learner_side)
        if record.policy in grouped[key]:
            return None
        grouped[key][record.policy] = record
    if not grouped or any(set(pair) != set(STYLE_POLICIES) for pair in grouped.values()):
        return None
    return np.asarray(
        [
            pair["aggressive"].engagement_rate - pair["strong_base"].engagement_rate
            for _, pair in sorted(grouped.items())
        ],
        dtype=np.float64,
    )


def _schedule_complete(records: Sequence[StyleEpisodeRecord], *, pairs_per_opponent: int) -> bool:
    expected_keys = len(DUEL_OPPONENTS) * pairs_per_opponent * 2
    for policy in STYLE_POLICIES:
        selected = [record for record in records if record.policy == policy]
        keys = {(record.opponent, record.pair_index, record.learner_side) for record in selected}
        if len(selected) != expected_keys or len(keys) != expected_keys:
            return False
        counts = Counter(record.opponent for record in selected)
        if counts != Counter({opponent: pairs_per_opponent * 2 for opponent in DUEL_OPPONENTS}):
            return False
        grouped: dict[tuple[str, int], set[str]] = defaultdict(set)
        for record in selected:
            grouped[(record.opponent, record.pair_index)].add(record.learner_side)
        if any(sides != {"host", "opponent"} for sides in grouped.values()):
            return False
    return True


def evaluate_aggressive_records(
    records: Sequence[StyleEpisodeRecord],
    *,
    expected_pairs_per_opponent: int,
    expected_scenario_hash: str,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> AggressiveEvaluationSummary:
    if expected_pairs_per_opponent <= 0:
        raise ValueError("Expected style evaluation pairs must be positive")
    expected_episodes = len(STYLE_POLICIES) * len(DUEL_OPPONENTS) * expected_pairs_per_opponent * 2
    identities = [
        (record.policy, record.opponent, record.pair_index, record.learner_side)
        for record in records
    ]
    protocol_inconsistencies = len(identities) - len(set(identities))
    protocol_inconsistencies += sum(
        int(
            record.protocol_inconsistent
            or record.action_tic_inconsistent
            or record.score_event_inconsistent
            or record.peer_tic_lag_max != 0
            or record.scenario_hash != expected_scenario_hash
        )
        for record in records
    )
    complete = len(records) == expected_episodes and _schedule_complete(
        records, pairs_per_opponent=expected_pairs_per_opponent
    )
    by_policy = {
        policy: [record for record in records if record.policy == policy]
        for policy in STYLE_POLICIES
    }
    policies = {
        policy: _policy_summary(selected) for policy, selected in by_policy.items() if selected
    }
    performance_available = set(policies) == set(STYLE_POLICIES)
    retention = (
        _safe_ratio(
            policies["aggressive"].performance,
            policies["strong_base"].performance,
        )
        if performance_available
        else 0.0
    )
    per_opponent_retention = {
        opponent: _safe_ratio(
            float(
                np.mean(
                    [
                        record.performance
                        for record in by_policy["aggressive"]
                        if record.opponent == opponent
                    ]
                )
            ),
            float(
                np.mean(
                    [
                        record.performance
                        for record in by_policy["strong_base"]
                        if record.opponent == opponent
                    ]
                )
            ),
        )
        for opponent in DUEL_OPPONENTS
        if all(
            any(record.opponent == opponent for record in by_policy[policy])
            for policy in STYLE_POLICIES
        )
    }
    differences = _paired_engagement_differences(records)
    engagement_ci = (
        paired_bootstrap_interval(
            differences,
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        )
        if differences is not None
        else None
    )
    engagement_delta = float(np.mean(differences)) if differences is not None else 0.0
    aggressive = policies.get("aggressive")
    gates = {
        "complete": complete,
        "protocol_clean": protocol_inconsistencies == 0,
        "skill_retention": retention + 1e-12 >= RETENTION_THRESHOLD,
        "per_opponent_retention": len(per_opponent_retention) == len(DUEL_OPPONENTS)
        and all(
            value + 1e-12 >= PER_OPPONENT_RETENTION_THRESHOLD
            for value in per_opponent_retention.values()
        ),
        "engagement_shift": engagement_ci is not None and engagement_ci[0] > 0,
        "valid_attack_rate": aggressive is not None
        and aggressive.valid_attack_rate + 1e-12 >= VALID_ATTACK_RATE_THRESHOLD,
        "objective_chase_controlled": aggressive is not None
        and aggressive.objective_chase_rate <= OBJECTIVE_CHASE_RATE_CEILING + 1e-12,
    }
    return AggressiveEvaluationSummary(
        complete=complete,
        passed=all(gates.values()),
        episodes=len(records),
        expected_episodes=expected_episodes,
        protocol_inconsistencies=protocol_inconsistencies,
        environment_retries=sum(record.environment_attempts - 1 for record in records),
        skill_retention=retention,
        per_opponent_retention=per_opponent_retention,
        engagement_initiation_delta=engagement_delta,
        engagement_initiation_delta_ci=engagement_ci,
        gates=gates,
        policies=policies,
    )


EnvironmentFactory = Callable[[DuelCase], SynchronousDuelEnv]


def run_style_episode(
    case: DuelCase,
    *,
    policy: EvaluationPolicy,
    graph: RegionGraph,
    config_path: Path,
    max_decisions: int,
    environment_factory: EnvironmentFactory | None = None,
) -> StyleEpisodeRecord:
    if case.split != "validation" or max_decisions <= 0:
        raise ValueError("Style episode requires validation and a positive limit")
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
    opponent_side = "opponent" if case.learner_side == "host" else "host"
    opponent = TeacherEvaluationPolicy(case.opponent, graph, side=opponent_side)
    action_tic_inconsistent = False
    peer_tic_lag_max = 0
    score_event_counts: Counter[str] = Counter()
    decisions = 0
    attacks = 0
    valid_hits = 0
    engagements = 0
    forward_hits = 0
    invalid_attacks = 0
    objective_chases = 0
    retreats = 0
    steps_since_hit = ENGAGEMENT_COOLDOWN + 1
    steps_since_received_hit = ENGAGEMENT_COOLDOWN + 1
    terminated = False
    truncated = False
    try:
        observations, reset_info = environment.reset()
        environment.set_shaping_scale(0.0)
        policy.reset(seed=case.seed ^ 0xA5A5A5A5)
        opponent.reset(seed=case.seed ^ 0x5A5A5A5A)
        learner_observation = (
            observations.host if case.learner_side == "host" else observations.opponent
        )
        initial_score = learner_observation.own_score
        initial_scores = {
            "host": observations.host.own_score,
            "opponent": observations.opponent.own_score,
        }
        while not (terminated or truncated):
            state = environment.teacher_state()
            learner_observation = (
                observations.host if case.learner_side == "host" else observations.opponent
            )
            learner_action = policy.act(learner_observation, state)
            opponent_action = opponent.act(
                observations.opponent if case.learner_side == "host" else observations.host,
                state,
            )
            host_action, opponent_action = (
                (learner_action, opponent_action)
                if case.learner_side == "host"
                else (opponent_action, learner_action)
            )
            step = environment.step(host_action, opponent_action)
            observations = type(observations)(step.host, step.opponent)
            decisions += 1
            terminated, truncated = step.terminated, step.truncated
            peer_tic_lag_max = max(peer_tic_lag_max, step.peer_tic_lag)
            score_event_counts.update(
                event.side for event in step.events if event.type is DuelEventType.SCORE
            )
            action_tic_inconsistent |= not valid_action_tic_boundary(
                step.action_tics,
                terminated=terminated,
                truncated=truncated,
            )
            hit_count = sum(
                event.side == case.learner_side and event.type is DuelEventType.VALID_HIT
                for event in step.events
            )
            received_hit = any(
                event.side != case.learner_side
                and event.side in ("host", "opponent")
                and event.type is DuelEventType.VALID_HIT
                for event in step.events
            )
            if learner_action in _ATTACK_ACTIONS:
                attacks += 1
                if hit_count:
                    valid_hits += hit_count
                    if steps_since_hit > ENGAGEMENT_COOLDOWN:
                        engagements += 1
                    if learner_action is MacroAction.FORWARD_ATTACK:
                        forward_hits += hit_count
                else:
                    invalid_attacks += 1
                    objective_chases += int(learner_observation.has_core)
            if (
                learner_action is MacroAction.MOVE_BACKWARD
                and steps_since_received_hit <= ENGAGEMENT_COOLDOWN
            ):
                retreats += 1
            steps_since_hit = 0 if hit_count else steps_since_hit + 1
            steps_since_received_hit = 0 if received_hit else steps_since_received_hit + 1
        learner_observation = (
            observations.host if case.learner_side == "host" else observations.opponent
        )
        final_scores = {
            "host": observations.host.own_score,
            "opponent": observations.opponent.own_score,
        }
        score_event_inconsistent = any(
            score_event_counts[side] != final_scores[side] - initial_scores[side]
            for side in ("host", "opponent")
        )
        learner_score = learner_observation.own_score
        opponent_score = learner_observation.opponent_score
        difference = learner_score - opponent_score
        outcome = "win" if difference > 0 else "loss" if difference < 0 else "draw"
        return StyleEpisodeRecord(
            policy=policy.name,
            split=case.split,
            opponent=case.opponent,
            pair_index=case.pair_index,
            seed=case.seed,
            learner_side=case.learner_side,
            outcome=outcome,
            objective_completed=learner_score > initial_score,
            learner_score=learner_score,
            opponent_score=opponent_score,
            decisions=decisions,
            attack_decisions=attacks,
            valid_hits=valid_hits,
            engagement_initiations=engagements,
            forward_hits=forward_hits,
            invalid_attack_decisions=invalid_attacks,
            objective_chase_decisions=objective_chases,
            retreat_decisions=retreats,
            terminated=terminated,
            truncated=truncated,
            peer_tic_lag_max=peer_tic_lag_max,
            protocol_inconsistent=False,
            action_tic_inconsistent=action_tic_inconsistent,
            score_event_inconsistent=score_event_inconsistent,
            scenario_hash=reset_info.scenario_hash,
        )
    finally:
        environment.close()
