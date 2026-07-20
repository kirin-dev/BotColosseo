from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from botcolosseo.agents.duel_teachers import create_duel_teacher
from botcolosseo.agents.model import AsymmetricActorCritic, RecurrentActor
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_protocol import DuelEventType
from botcolosseo.envs.duel_types import DuelActorObservation, DuelPrivilegedState
from botcolosseo.envs.synchronous_duel import SynchronousDuelEnv
from botcolosseo.evaluation.m1 import wilson_interval
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS, DuelCase
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.training.duel_rollout import actor_observation_tensors

M2_POLICIES = ("ppo", "bc", "random_legal")
FROZEN_M2_THRESHOLDS = {
    "ppo_win_rate_minus_bc": 0.10,
    "ppo_win_rate_minus_random": 0.20,
    "ppo_objective_rate_minus_bc": 0.10,
    "paired_score_lcb": 0.0,
    "per_opponent_ppo_minus_bc_floor": -0.05,
}


class EvaluationPolicy(Protocol):
    name: str

    def reset(self, *, seed: int) -> None: ...

    def act(
        self, observation: DuelActorObservation, state: DuelPrivilegedState
    ) -> MacroAction: ...


class TeacherEvaluationPolicy:
    def __init__(self, name: str, graph: RegionGraph, *, side: str) -> None:
        self.name = name
        self._teacher = create_duel_teacher(name, graph, side=side)

    def reset(self, *, seed: int) -> None:
        self._teacher.reset(seed=seed)

    def act(
        self, observation: DuelActorObservation, state: DuelPrivilegedState
    ) -> MacroAction:
        del observation
        return MacroAction(self._teacher.act(state))


class GreedyActorPolicy:
    def __init__(
        self,
        name: str,
        actor: RecurrentActor,
        *,
        device: torch.device,
    ) -> None:
        self.name = name
        self._actor = actor.to(device).eval()
        self._device = device
        self._hidden: torch.Tensor | None = None
        self._episode_start = True

    def reset(self, *, seed: int) -> None:
        del seed
        self._hidden = self._actor.initial_state(1, device=self._device)
        self._episode_start = True

    @torch.no_grad()
    def act(
        self, observation: DuelActorObservation, state: DuelPrivilegedState
    ) -> MacroAction:
        del state
        if self._hidden is None:
            raise RuntimeError("Evaluation policy must be reset before act")
        inputs = actor_observation_tensors(
            observation,
            episode_start=self._episode_start,
            device=self._device,
        )
        output = self._actor(*inputs, self._hidden)
        self._hidden = output.hidden
        self._episode_start = False
        return MacroAction(int(output.logits[0, 0].argmax()))


def load_actor_policy(
    name: str,
    checkpoint: Path,
    *,
    device: torch.device,
    expected_scenario_hash: str,
) -> GreedyActorPolicy:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported learned-policy checkpoint schema")
    metadata = payload.get("metadata", {})
    if metadata.get("scenario_hash") != expected_scenario_hash:
        raise ValueError(f"{name} checkpoint scenario hash does not match")
    if name == "bc":
        actor = RecurrentActor()
        actor.load_state_dict(payload["model"])
    elif name == "ppo":
        model = AsymmetricActorCritic()
        model.load_state_dict(payload["model"])
        actor = model.actor
    else:
        raise ValueError(f"Unsupported learned policy: {name}")
    return GreedyActorPolicy(name, actor, device=device)


@dataclass(frozen=True)
class M2EpisodeRecord:
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
    terminated: bool
    truncated: bool
    peer_tic_lag_max: int
    protocol_inconsistent: bool
    scenario_hash: str
    action_tic_inconsistent: bool = False
    score_event_inconsistent: bool = False
    environment_attempts: int = 1

    @property
    def score_difference(self) -> int:
        return self.learner_score - self.opponent_score

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "score_difference": self.score_difference}


@dataclass(frozen=True)
class RateSummary:
    successes: int
    trials: int
    rate: float
    wilson_lower: float
    wilson_upper: float


@dataclass(frozen=True)
class OpponentSummary:
    episodes: int
    wins: RateSummary
    draws: RateSummary
    objectives: RateSummary
    mean_score_difference: float


@dataclass(frozen=True)
class PolicySummary:
    episodes: int
    wins: RateSummary
    draws: RateSummary
    objectives: RateSummary
    mean_score_difference: float
    opponents: dict[str, OpponentSummary]


@dataclass(frozen=True)
class M2EvaluationSummary:
    official: bool
    complete: bool
    passed: bool
    episodes: int
    expected_episodes: int
    protocol_inconsistencies: int
    protocol_counts: dict[str, int]
    artifact_inconsistencies: int
    environment_retries: int
    paired_score_difference_ci: tuple[float, float] | None
    gates: dict[str, bool]
    policies: dict[str, PolicySummary]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_paired_schedule(
    cases: Sequence[DuelCase],
    *,
    expected_split: str,
    pairs_per_opponent: int,
) -> None:
    if pairs_per_opponent <= 0:
        raise ValueError("pairs_per_opponent must be positive")
    expected_rows = len(DUEL_OPPONENTS) * pairs_per_opponent * 2
    if len(cases) != expected_rows:
        raise ValueError(f"Expected {expected_rows} paired cases, found {len(cases)}")
    if any(case.split != expected_split for case in cases):
        raise ValueError(f"Schedule contains rows outside split {expected_split}")
    for first, second in zip(cases[::2], cases[1::2], strict=True):
        if (
            first.opponent != second.opponent
            or first.pair_index != second.pair_index
            or first.seed != second.seed
            or (first.learner_side, second.learner_side) != ("host", "opponent")
        ):
            raise ValueError("Schedule must replay adjacent host/opponent side-swapped pairs")
    counts: Counter[str] = Counter()
    seen: set[tuple[str, int, str]] = set()
    grouped: dict[tuple[str, int], list[DuelCase]] = defaultdict(list)
    for case in cases:
        if case.opponent not in DUEL_OPPONENTS:
            raise ValueError(f"Unknown opponent in schedule: {case.opponent}")
        identity = (case.opponent, case.pair_index, case.learner_side)
        if identity in seen:
            raise ValueError(f"Pair is not side-swapped exactly once: {identity}")
        seen.add(identity)
        grouped[(case.opponent, case.pair_index)].append(case)
    for (opponent, _), pair in grouped.items():
        if len(pair) != 2 or {case.learner_side for case in pair} != {
            "host",
            "opponent",
        }:
            raise ValueError("Every pair must be side-swapped exactly once")
        if len({case.seed for case in pair}) != 1:
            raise ValueError("Side-swapped cases must share a seed")
        counts[opponent] += 1
    expected_counts = Counter({opponent: pairs_per_opponent for opponent in DUEL_OPPONENTS})
    if counts != expected_counts:
        raise ValueError("Schedule is not balanced across opponents")


def load_duel_cases(
    path: Path, *, expected_split: str, pairs_per_opponent: int
) -> tuple[DuelCase, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = tuple(DuelCase(**item) for item in payload)
    validate_paired_schedule(
        cases,
        expected_split=expected_split,
        pairs_per_opponent=pairs_per_opponent,
    )
    return cases


def paired_bootstrap_interval(
    differences: np.ndarray | Sequence[float],
    *,
    seed: int,
    samples: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("differences must be a non-empty vector")
    if samples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(samples, values.size))
    means = values[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, (tail, 1.0 - tail))
    return float(lower), float(upper)


def _rate(successes: int, trials: int) -> RateSummary:
    lower, upper = wilson_interval(successes, trials)
    return RateSummary(successes, trials, successes / trials, lower, upper)


def _opponent_summary(records: Sequence[M2EpisodeRecord]) -> OpponentSummary:
    trials = len(records)
    return OpponentSummary(
        episodes=trials,
        wins=_rate(sum(record.outcome == "win" for record in records), trials),
        draws=_rate(sum(record.outcome == "draw" for record in records), trials),
        objectives=_rate(sum(record.objective_completed for record in records), trials),
        mean_score_difference=float(
            np.mean([record.score_difference for record in records])
        ),
    )


def _policy_summary(records: Sequence[M2EpisodeRecord]) -> PolicySummary:
    trials = len(records)
    opponents = {
        opponent: _opponent_summary(
            [record for record in records if record.opponent == opponent]
        )
        for opponent in DUEL_OPPONENTS
        if any(record.opponent == opponent for record in records)
    }
    return PolicySummary(
        episodes=trials,
        wins=_rate(sum(record.outcome == "win" for record in records), trials),
        draws=_rate(sum(record.outcome == "draw" for record in records), trials),
        objectives=_rate(sum(record.objective_completed for record in records), trials),
        mean_score_difference=float(
            np.mean([record.score_difference for record in records])
        ),
        opponents=opponents,
    )


def _paired_score_differences(
    records: Sequence[M2EpisodeRecord],
) -> np.ndarray | None:
    scores: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for record in records:
        if record.environment_attempts <= 0:
            raise ValueError("environment_attempts must be positive")
        if record.policy in ("ppo", "bc"):
            scores[(record.policy, record.pair_index, record.opponent)].append(
                record.score_difference
            )
    pair_keys = {
        (record.pair_index, record.opponent)
        for record in records
        if record.policy == "ppo"
    }
    differences: list[float] = []
    for pair_index, opponent in sorted(pair_keys):
        ppo = scores.get(("ppo", pair_index, opponent), [])
        bc = scores.get(("bc", pair_index, opponent), [])
        if len(ppo) != 2 or len(bc) != 2:
            return None
        differences.append(float(np.mean(ppo) - np.mean(bc)))
    return np.asarray(differences) if differences else None


def _at_least(value: float, threshold: float) -> bool:
    return value + 1e-12 >= threshold


def valid_action_tic_boundary(
    action_tics: int, *, terminated: bool, truncated: bool
) -> bool:
    return action_tics == 4 or (
        (terminated or truncated) and 0 <= action_tics < 4
    )


def _records_are_paired(
    by_policy: dict[str, list[M2EpisodeRecord]], *, pairs_per_opponent: int
) -> bool:
    policy_keys: list[set[tuple[str, int, str]]] = []
    for policy in M2_POLICIES:
        grouped: dict[tuple[str, int], list[M2EpisodeRecord]] = defaultdict(list)
        for record in by_policy[policy]:
            grouped[(record.opponent, record.pair_index)].append(record)
        if len(grouped) != len(DUEL_OPPONENTS) * pairs_per_opponent:
            return False
        if any(
            len(pair) != 2
            or {record.learner_side for record in pair} != {"host", "opponent"}
            or len({record.seed for record in pair}) != 1
            for pair in grouped.values()
        ):
            return False
        policy_keys.append(
            {
                (record.opponent, record.pair_index, record.learner_side)
                for record in by_policy[policy]
            }
        )
    return policy_keys[0] == policy_keys[1] == policy_keys[2]


def evaluate_m2_records(
    records: Sequence[M2EpisodeRecord],
    *,
    official: bool,
    expected_pairs_per_opponent: int,
    bootstrap_seed: int,
    bootstrap_samples: int,
    expected_scenario_hash: str,
    bootstrap_confidence: float = 0.95,
    artifact_inconsistencies: int = 0,
) -> M2EvaluationSummary:
    if artifact_inconsistencies < 0:
        raise ValueError("artifact_inconsistencies must be nonnegative")
    expected_per_policy = len(DUEL_OPPONENTS) * expected_pairs_per_opponent * 2
    expected_episodes = expected_per_policy * len(M2_POLICIES)
    identities = [
        (record.policy, record.opponent, record.pair_index, record.learner_side)
        for record in records
    ]
    protocol_counts: Counter[str] = Counter()
    protocol_counts["duplicate_rows"] = len(identities) - len(set(identities))
    for record in records:
        score_sign = (record.score_difference > 0) - (record.score_difference < 0)
        expected_outcome = {1: "win", 0: "draw", -1: "loss"}[score_sign]
        protocol_counts["explicit_inconsistency_rows"] += int(
            record.protocol_inconsistent
        )
        protocol_counts["action_tic_inconsistency_rows"] += int(
            record.action_tic_inconsistent
        )
        protocol_counts["score_event_inconsistency_rows"] += int(
            record.score_event_inconsistent
        )
        protocol_counts["peer_tic_lag_rows"] += int(record.peer_tic_lag_max != 0)
        protocol_counts["scenario_mismatch_rows"] += int(
            record.scenario_hash != expected_scenario_hash
        )
        protocol_counts["outcome_score_mismatch_rows"] += int(
            record.outcome != expected_outcome
        )
        protocol_counts["fairness_schema_rows"] += int(
            record.learner_side not in ("host", "opponent")
            or record.opponent not in DUEL_OPPONENTS
        )
        protocol_counts["invalid_terminal_boundary_rows"] += int(
            record.terminated == record.truncated
        )
        protocol_counts["terminated_rows"] += int(record.terminated)
        protocol_counts["truncated_rows"] += int(record.truncated)
        protocol_counts["environment_retry_rows"] += int(
            record.environment_attempts > 1
        )
    issue_names = (
        "duplicate_rows",
        "explicit_inconsistency_rows",
        "action_tic_inconsistency_rows",
        "score_event_inconsistency_rows",
        "peer_tic_lag_rows",
        "scenario_mismatch_rows",
        "outcome_score_mismatch_rows",
        "fairness_schema_rows",
        "invalid_terminal_boundary_rows",
    )
    protocol_inconsistencies = sum(protocol_counts[name] for name in issue_names)

    by_policy = {
        policy: [record for record in records if record.policy == policy]
        for policy in M2_POLICIES
    }
    counts_complete = len(records) == expected_episodes and all(
        len(policy_records) == expected_per_policy
        for policy_records in by_policy.values()
    )
    balance_complete = counts_complete and all(
        sum(record.opponent == opponent for record in by_policy[policy])
        == expected_pairs_per_opponent * 2
        for policy in M2_POLICIES
        for opponent in DUEL_OPPONENTS
    )
    policies = {
        policy: _policy_summary(policy_records)
        for policy, policy_records in by_policy.items()
        if policy_records
    }
    differences = _paired_score_differences(records)
    paired_ci = (
        paired_bootstrap_interval(
            differences,
            seed=bootstrap_seed,
            samples=bootstrap_samples,
            confidence=bootstrap_confidence,
        )
        if differences is not None
        else None
    )

    performance_available = set(policies) == set(M2_POLICIES)
    win_vs_bc = performance_available and (
        _at_least(
            policies["ppo"].wins.rate - policies["bc"].wins.rate,
            FROZEN_M2_THRESHOLDS["ppo_win_rate_minus_bc"],
        )
    )
    win_vs_random = performance_available and (
        _at_least(
            policies["ppo"].wins.rate - policies["random_legal"].wins.rate,
            FROZEN_M2_THRESHOLDS["ppo_win_rate_minus_random"],
        )
    )
    objective_vs_bc = performance_available and (
        _at_least(
            policies["ppo"].objectives.rate - policies["bc"].objectives.rate,
            FROZEN_M2_THRESHOLDS["ppo_objective_rate_minus_bc"],
        )
    )
    paired_positive = (
        paired_ci is not None
        and paired_ci[0] > FROZEN_M2_THRESHOLDS["paired_score_lcb"]
    )
    opponent_floor = performance_available and all(
        _at_least(
            policies["ppo"].opponents[opponent].wins.rate
            - policies["bc"].opponents[opponent].wins.rate,
            FROZEN_M2_THRESHOLDS["per_opponent_ppo_minus_bc_floor"],
        )
        for opponent in DUEL_OPPONENTS
        if opponent in policies["ppo"].opponents
        and opponent in policies["bc"].opponents
    ) and all(
        opponent in policies["ppo"].opponents and opponent in policies["bc"].opponents
        for opponent in DUEL_OPPONENTS
    )
    complete = (
        balance_complete
        and set(record.policy for record in records) == set(M2_POLICIES)
        and _records_are_paired(
            by_policy, pairs_per_opponent=expected_pairs_per_opponent
        )
        and (not official or {record.split for record in records} == {"test"})
    )
    gates = {
        "official": official,
        "complete": complete,
        "protocol_clean": protocol_inconsistencies == 0,
        "artifact_clean": artifact_inconsistencies == 0,
        "ppo_win_rate_minus_bc": bool(win_vs_bc),
        "ppo_win_rate_minus_random": bool(win_vs_random),
        "ppo_objective_rate_minus_bc": bool(objective_vs_bc),
        "paired_score_lcb_positive": bool(paired_positive),
        "per_opponent_floor": bool(opponent_floor),
    }
    return M2EvaluationSummary(
        official=official,
        complete=complete,
        passed=all(gates.values()),
        episodes=len(records),
        expected_episodes=expected_episodes,
        protocol_inconsistencies=protocol_inconsistencies,
        protocol_counts=dict(sorted(protocol_counts.items())),
        artifact_inconsistencies=artifact_inconsistencies,
        environment_retries=sum(record.environment_attempts - 1 for record in records),
        paired_score_difference_ci=paired_ci,
        gates=gates,
        policies=policies,
    )


EnvironmentFactory = Callable[[DuelCase], SynchronousDuelEnv]


def run_m2_episode(
    case: DuelCase,
    *,
    policy: EvaluationPolicy,
    graph: RegionGraph,
    config_path: Path,
    max_decisions: int,
    environment_factory: EnvironmentFactory | None = None,
) -> M2EpisodeRecord:
    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
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
    terminated = False
    truncated = False
    try:
        observations, reset_info = environment.reset()
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
            state = environment.teacher_state()
            learner_observation = (
                observations.host
                if case.learner_side == "host"
                else observations.opponent
            )
            learner_action = policy.act(learner_observation, state)
            opponent_action = opponent.act(
                observations.opponent
                if case.learner_side == "host"
                else observations.host,
                state,
            )
            host_action, away_action = (
                (learner_action, opponent_action)
                if case.learner_side == "host"
                else (opponent_action, learner_action)
            )
            step = environment.step(host_action, away_action)
            observations = type(observations)(step.host, step.opponent)
            decisions += 1
            terminated, truncated = step.terminated, step.truncated
            peer_tic_lag_max = max(peer_tic_lag_max, step.peer_tic_lag)
            score_event_counts.update(
                event.side
                for event in step.events
                if event.type is DuelEventType.SCORE
            )
            action_tic_inconsistent |= not valid_action_tic_boundary(
                step.action_tics,
                terminated=step.terminated,
                truncated=step.truncated,
            )
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
        score_difference = learner_score - opponent_score
        outcome = "win" if score_difference > 0 else "loss" if score_difference < 0 else "draw"
        return M2EpisodeRecord(
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
            terminated=terminated,
            truncated=truncated,
            peer_tic_lag_max=peer_tic_lag_max,
            protocol_inconsistent=False,
            scenario_hash=reset_info.scenario_hash,
            action_tic_inconsistent=action_tic_inconsistent,
            score_event_inconsistent=score_event_inconsistent,
        )
    finally:
        environment.close()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_m2_evidence(
    output_dir: Path,
    *,
    records: Sequence[M2EpisodeRecord],
    summary: M2EvaluationSummary,
    manifest: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [record.to_dict() for record in records]
    csv_path = output_dir / "episodes.csv"
    csv_temp = output_dir / ".episodes.csv.tmp"
    fields = list(rows[0]) if rows else list(M2EpisodeRecord.__dataclass_fields__) + [
        "score_difference"
    ]
    with csv_temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    csv_temp.replace(csv_path)
    summary_path = output_dir / "summary.json"
    summary_temp = output_dir / ".summary.json.tmp"
    summary_temp.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_temp.replace(summary_path)
    complete_manifest = {
        **manifest,
        "episodes_sha256": sha256_file(csv_path),
        "summary_sha256": sha256_file(summary_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_temp = output_dir / ".manifest.json.tmp"
    manifest_temp.write_text(
        json.dumps(complete_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_temp.replace(manifest_path)
