from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from botcolosseo.envs.defensive_signals import (
    defensive_risk,
    in_defensive_half,
    in_protective_zone,
    opponent_carrier_id,
    unnecessary_guard,
)
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

DEFENSIVE_POLICIES = ("strong_base", "defensive")


@dataclass(frozen=True)
class DefensiveEpisodeRecord:
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
    risk_decisions: int
    protective_presence_decisions: int
    carrier_opportunities: int
    carrier_denials: int
    recovery_opportunities: int
    recoveries: int
    no_risk_decisions: int
    unnecessary_guard_decisions: int
    low_health_opportunities: int
    successful_escapes: int
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
            self.risk_decisions,
            self.protective_presence_decisions,
            self.carrier_opportunities,
            self.carrier_denials,
            self.recovery_opportunities,
            self.recoveries,
            self.no_risk_decisions,
            self.unnecessary_guard_decisions,
            self.low_health_opportunities,
            self.successful_escapes,
            self.peer_tic_lag_max,
        )
        if self.policy not in DEFENSIVE_POLICIES:
            raise ValueError("Unknown Defensive evaluation policy")
        if self.split != "validation" or self.opponent not in DUEL_OPPONENTS:
            raise ValueError("Defensive evaluation requires frozen validation scripts")
        if self.learner_side not in ("host", "opponent"):
            raise ValueError("Defensive evaluation learner side is invalid")
        if any(value < 0 for value in counters) or self.environment_attempts <= 0:
            raise ValueError("Defensive evaluation counters must be nonnegative")
        if self.decisions <= 0 or self.risk_decisions + self.no_risk_decisions != self.decisions:
            raise ValueError("Defensive evaluation decision accounting is invalid")
        if self.protective_presence_decisions > self.risk_decisions:
            raise ValueError("Protective presence exceeds risk decisions")
        if self.unnecessary_guard_decisions > self.no_risk_decisions:
            raise ValueError("Unnecessary guards exceed no-risk decisions")
        difference = self.score_difference
        expected = "win" if difference > 0 else "loss" if difference < 0 else "draw"
        if self.outcome != expected:
            raise ValueError("Defensive evaluation outcome does not match score")
        if self.terminated == self.truncated or not self.scenario_hash:
            raise ValueError("Defensive evaluation terminal or scenario metadata is invalid")

    @property
    def score_difference(self) -> int:
        return self.learner_score - self.opponent_score

    @property
    def performance(self) -> float:
        points = {"win": 1.0, "draw": 0.5, "loss": 0.0}[self.outcome]
        return 0.5 * (points + float(self.objective_completed))

    @property
    def protective_presence_rate(self) -> float:
        return (
            self.protective_presence_decisions / self.risk_decisions
            if self.risk_decisions
            else 0.0
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "score_difference": self.score_difference,
            "performance": self.performance,
            "protective_presence_rate": self.protective_presence_rate,
        }


@dataclass(frozen=True)
class DefensivePolicySummary:
    episodes: int
    win_rate: float
    objective_rate: float
    mean_score_difference: float
    performance: float
    risk_decisions: int
    protective_presence_rate: float
    carrier_opportunities: int
    carrier_denials: int
    recovery_opportunities: int
    recoveries: int
    denial_recovery_rate: float
    unnecessary_guard_rate: float
    low_health_opportunities: int
    successful_escapes: int


@dataclass(frozen=True)
class DefensiveEvaluationSummary:
    complete: bool
    passed: bool
    episodes: int
    expected_episodes: int
    protocol_inconsistencies: int
    environment_retries: int
    skill_retention: float
    per_opponent_retention: dict[str, float]
    protective_presence_delta: float
    protective_presence_delta_ci: tuple[float, float] | None
    gates: dict[str, bool]
    policies: dict[str, DefensivePolicySummary]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _retention(style: float, base: float) -> float:
    return style / base if base > 1e-12 else float(style + 1e-12 >= base)


def _policy_summary(records: Sequence[DefensiveEpisodeRecord]) -> DefensivePolicySummary:
    episodes = len(records)
    risk = sum(row.risk_decisions for row in records)
    carrier = sum(row.carrier_opportunities for row in records)
    recovery = sum(row.recovery_opportunities for row in records)
    denials = sum(row.carrier_denials for row in records)
    recoveries = sum(row.recoveries for row in records)
    no_risk = sum(row.no_risk_decisions for row in records)
    low_health = sum(row.low_health_opportunities for row in records)
    return DefensivePolicySummary(
        episodes=episodes,
        win_rate=sum(row.outcome == "win" for row in records) / episodes,
        objective_rate=sum(row.objective_completed for row in records) / episodes,
        mean_score_difference=float(np.mean([row.score_difference for row in records])),
        performance=float(np.mean([row.performance for row in records])),
        risk_decisions=risk,
        protective_presence_rate=_ratio(
            sum(row.protective_presence_decisions for row in records), risk
        ),
        carrier_opportunities=carrier,
        carrier_denials=denials,
        recovery_opportunities=recovery,
        recoveries=recoveries,
        denial_recovery_rate=_ratio(denials + recoveries, carrier + recovery),
        unnecessary_guard_rate=_ratio(
            sum(row.unnecessary_guard_decisions for row in records), no_risk
        ),
        low_health_opportunities=low_health,
        successful_escapes=sum(row.successful_escapes for row in records),
    )


def _schedule_complete(
    records: Sequence[DefensiveEpisodeRecord], *, pairs_per_opponent: int
) -> bool:
    expected = len(DUEL_OPPONENTS) * pairs_per_opponent * 2
    for policy in DEFENSIVE_POLICIES:
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


def evaluate_defensive_records(
    records: Sequence[DefensiveEpisodeRecord],
    *,
    expected_pairs_per_opponent: int,
    expected_scenario_hash: str,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> DefensiveEvaluationSummary:
    if expected_pairs_per_opponent <= 0:
        raise ValueError("Expected Defensive evaluation pairs must be positive")
    expected = len(DEFENSIVE_POLICIES) * len(DUEL_OPPONENTS) * expected_pairs_per_opponent * 2
    identities = [
        (row.policy, row.opponent, row.pair_index, row.learner_side) for row in records
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
        for policy in DEFENSIVE_POLICIES
    }
    policies = {
        policy: _policy_summary(rows) for policy, rows in by_policy.items() if rows
    }
    available = set(policies) == set(DEFENSIVE_POLICIES)
    retention = (
        _retention(policies["defensive"].performance, policies["strong_base"].performance)
        if available
        else 0.0
    )
    per_opponent = {
        opponent: _retention(
            float(
                np.mean(
                    [
                        row.performance
                        for row in by_policy["defensive"]
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
        if all(any(row.opponent == opponent for row in by_policy[p]) for p in DEFENSIVE_POLICIES)
    }
    grouped: dict[tuple[str, int, str], dict[str, DefensiveEpisodeRecord]] = defaultdict(dict)
    for row in records:
        grouped[(row.opponent, row.pair_index, row.learner_side)][row.policy] = row
    differences = None
    if grouped and all(set(pair) == set(DEFENSIVE_POLICIES) for pair in grouped.values()):
        differences = np.asarray(
            [
                pair["defensive"].protective_presence_rate
                - pair["strong_base"].protective_presence_rate
                for _, pair in sorted(grouped.items())
            ],
            dtype=np.float64,
        )
    interval = (
        paired_bootstrap_interval(differences, seed=bootstrap_seed, samples=bootstrap_samples)
        if differences is not None
        else None
    )
    delta = float(np.mean(differences)) if differences is not None else 0.0
    base = policies.get("strong_base")
    defensive = policies.get("defensive")
    gates = {
        "complete": complete,
        "protocol_clean": inconsistencies == 0,
        "skill_retention": retention + 1e-12 >= 0.85,
        "per_opponent_retention": len(per_opponent) == len(DUEL_OPPONENTS)
        and all(value + 1e-12 >= 0.75 for value in per_opponent.values()),
        "protective_presence_shift": interval is not None and interval[0] > 0,
        "denial_recovery_improved": base is not None
        and defensive is not None
        and defensive.carrier_opportunities + defensive.recovery_opportunities > 0
        and defensive.denial_recovery_rate > base.denial_recovery_rate,
        "unnecessary_guard_controlled": defensive is not None
        and defensive.unnecessary_guard_rate <= 0.20 + 1e-12,
        "objective_retention": base is not None
        and defensive is not None
        and defensive.objective_rate + 1e-12 >= 0.85 * base.objective_rate,
    }
    return DefensiveEvaluationSummary(
        complete=complete,
        passed=all(gates.values()),
        episodes=len(records),
        expected_episodes=expected,
        protocol_inconsistencies=inconsistencies,
        environment_retries=sum(row.environment_attempts - 1 for row in records),
        skill_retention=retention,
        per_opponent_retention=per_opponent,
        protective_presence_delta=delta,
        protective_presence_delta_ci=interval,
        gates=gates,
        policies=policies,
    )


EnvironmentFactory = Callable[[DuelCase], SynchronousDuelEnv]


def run_defensive_episode(
    case: DuelCase,
    *,
    policy: EvaluationPolicy,
    graph: RegionGraph,
    config_path: Path,
    max_decisions: int,
    environment_factory: EnvironmentFactory | None = None,
) -> DefensiveEpisodeRecord:
    if case.split != "validation" or max_decisions <= 0:
        raise ValueError("Defensive episode requires validation and a positive limit")
    env = environment_factory(case) if environment_factory else SynchronousDuelEnv(
        config_path=config_path, region_graph=graph, seed=case.seed, max_decisions=max_decisions
    )
    opponent_side = "opponent" if case.learner_side == "host" else "host"
    opponent = TeacherEvaluationPolicy(case.opponent, graph, side=opponent_side)
    counts: Counter[str] = Counter()
    score_events: Counter[str] = Counter()
    action_tic_inconsistent = False
    peer_tic_lag_max = 0
    terminated = truncated = False
    low_health_active = False
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
        while not (terminated or truncated):
            state = env.teacher_state()
            risk = defensive_risk(state, case.learner_side)
            counts["decisions"] += 1
            counts["risk"] += int(risk)
            counts["no_risk"] += int(not risk)
            counts["presence"] += int(risk and in_protective_zone(state, case.learner_side))
            carrier_opportunity = state.carrier == opponent_carrier_id(case.learner_side)
            recovery_opportunity = state.carrier == 0 and in_defensive_half(
                state.core_x, case.learner_side
            )
            counts["carrier_opportunities"] += int(carrier_opportunity)
            counts["recovery_opportunities"] += int(recovery_opportunity)
            counts["unnecessary_guard"] += int(unnecessary_guard(state, case.learner_side))
            own_health, opposing_health = (
                (state.host_health, state.opponent_health)
                if case.learner_side == "host"
                else (state.opponent_health, state.host_health)
            )
            low_health = own_health <= 25.0 and opposing_health > own_health
            counts["low_health_opportunities"] += int(low_health)
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
            next_state = env.teacher_state()
            observations = type(observations)(step.host, step.opponent)
            terminated, truncated = step.terminated, step.truncated
            peer_tic_lag_max = max(peer_tic_lag_max, step.peer_tic_lag)
            score_events.update(
                event.side
                for event in step.events
                if event.type is DuelEventType.SCORE
            )
            action_tic_inconsistent |= not valid_action_tic_boundary(
                step.action_tics, terminated=terminated, truncated=truncated
            )
            hit = any(
                event.side == case.learner_side
                and event.type is DuelEventType.VALID_HIT
                for event in step.events
            )
            drop = any(
                event.side == opponent_side and event.type is DuelEventType.DROP
                for event in step.events
            )
            pickup = any(
                event.side == case.learner_side
                and event.type is DuelEventType.PICKUP
                for event in step.events
            )
            counts["carrier_denials"] += int(carrier_opportunity and hit and drop)
            counts["recoveries"] += int(recovery_opportunity and pickup)
            next_own_health = (
                next_state.host_health
                if case.learner_side == "host"
                else next_state.opponent_health
            )
            if low_health and not low_health_active:
                low_health_active = True
            if low_health_active and not low_health and next_own_health > 0:
                counts["successful_escapes"] += 1
                low_health_active = False
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
        difference = learner_score - opponent_score
        return DefensiveEpisodeRecord(
            policy=policy.name,
            split=case.split,
            opponent=case.opponent,
            pair_index=case.pair_index,
            seed=case.seed,
            learner_side=case.learner_side,
            outcome="win" if difference > 0 else "loss" if difference < 0 else "draw",
            objective_completed=learner_score > initial_score,
            learner_score=learner_score,
            opponent_score=opponent_score,
            decisions=counts["decisions"],
            risk_decisions=counts["risk"],
            protective_presence_decisions=counts["presence"],
            carrier_opportunities=counts["carrier_opportunities"],
            carrier_denials=counts["carrier_denials"],
            recovery_opportunities=counts["recovery_opportunities"],
            recoveries=counts["recoveries"],
            no_risk_decisions=counts["no_risk"],
            unnecessary_guard_decisions=counts["unnecessary_guard"],
            low_health_opportunities=counts["low_health_opportunities"],
            successful_escapes=counts["successful_escapes"],
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
