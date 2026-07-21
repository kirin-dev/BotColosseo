from __future__ import annotations

import csv
import math
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement
from pathlib import Path

from botcolosseo.agents.league_opponents import OpponentSpec
from botcolosseo.envs.duel_protocol import DuelEventType
from botcolosseo.envs.synchronous_duel import DuelObservations, SynchronousDuelEnv
from botcolosseo.evaluation.m1 import wilson_interval
from botcolosseo.evaluation.m2 import valid_action_tic_boundary
from botcolosseo.scenarios.league_splits import LeagueCase
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.training.duel_rollout import DuelOpponentController


@dataclass(frozen=True)
class CrossplayRow:
    left_policy: str
    right_policy: str
    split: str
    pair_index: int
    seed: int
    left_side: str
    outcome: str
    left_objective_completed: bool
    right_objective_completed: bool
    left_score: int
    right_score: int
    decisions: int
    terminated: bool
    truncated: bool
    peer_tic_lag_max: int
    protocol_inconsistent: bool
    action_tic_inconsistent: bool
    score_event_inconsistent: bool
    scenario_hash: str
    environment_attempts: int

    def __post_init__(self) -> None:
        if not self.left_policy or not self.right_policy:
            raise ValueError("Cross-play policy IDs must be non-empty")
        if self.split != "validation" or self.left_side not in ("host", "opponent"):
            raise ValueError("Cross-play rows require a validation side")
        if min(
            self.pair_index,
            self.seed,
            self.left_score,
            self.right_score,
            self.decisions,
            self.peer_tic_lag_max,
        ) < 0 or self.environment_attempts <= 0:
            raise ValueError("Cross-play counters must be nonnegative")
        difference = self.left_score - self.right_score
        expected = "win" if difference > 0 else "loss" if difference < 0 else "draw"
        if self.outcome != expected:
            raise ValueError("Cross-play outcome does not match score")
        if self.terminated == self.truncated:
            raise ValueError("Cross-play row must have one terminal boundary")
        if not self.scenario_hash:
            raise ValueError("Cross-play row requires a scenario hash")

    @property
    def score_difference(self) -> int:
        return self.left_score - self.right_score

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "score_difference": self.score_difference}


EpisodeRunner = Callable[[OpponentSpec, OpponentSpec, LeagueCase], CrossplayRow]
EnvironmentFactory = Callable[[LeagueCase], SynchronousDuelEnv]


def run_crossplay_episode(
    case: LeagueCase,
    *,
    left_spec: OpponentSpec,
    right_spec: OpponentSpec,
    left_controller: DuelOpponentController,
    right_controller: DuelOpponentController,
    graph: RegionGraph,
    config_path: Path,
    max_decisions: int,
    environment_factory: EnvironmentFactory | None = None,
) -> CrossplayRow:
    if case.split != "validation" or max_decisions <= 0:
        raise ValueError("Cross-play episode requires a validation case and positive limit")
    if left_spec.scenario_hash != right_spec.scenario_hash:
        raise ValueError("Cross-play policy scenario hashes do not match")
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
    peer_tic_lag_max = 0
    score_event_counts: Counter[str] = Counter()
    decisions = 0
    terminated = False
    truncated = False
    try:
        observations, reset_info = environment.reset()
        environment.set_shaping_scale(0.0)
        left_controller.reset(seed=case.seed ^ 0xA5A5A5A5)
        right_controller.reset(seed=case.seed ^ 0x5A5A5A5A)
        initial_scores = {
            "host": observations.host.own_score,
            "opponent": observations.opponent.own_score,
        }
        while not (terminated or truncated):
            left_observation = (
                observations.host
                if case.learner_side == "host"
                else observations.opponent
            )
            right_observation = (
                observations.opponent
                if case.learner_side == "host"
                else observations.host
            )
            left_action = left_controller.act(
                left_observation, environment.teacher_state
            )
            right_action = right_controller.act(
                right_observation, environment.teacher_state
            )
            host_action, opponent_action = (
                (left_action, right_action)
                if case.learner_side == "host"
                else (right_action, left_action)
            )
            step = environment.step(host_action, opponent_action)
            observations = DuelObservations(step.host, step.opponent)
            decisions += 1
            if decisions > max_decisions:
                raise RuntimeError("Cross-play environment exceeded max_decisions")
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
        final_scores = {
            "host": observations.host.own_score,
            "opponent": observations.opponent.own_score,
        }
        score_event_inconsistent = any(
            score_event_counts[side] != final_scores[side] - initial_scores[side]
            for side in ("host", "opponent")
        )
        left_side = case.learner_side
        right_side = "opponent" if left_side == "host" else "host"
        left_score = final_scores[left_side]
        right_score = final_scores[right_side]
        difference = left_score - right_score
        outcome = "win" if difference > 0 else "loss" if difference < 0 else "draw"
        return CrossplayRow(
            left_policy=left_spec.opponent_id,
            right_policy=right_spec.opponent_id,
            split=case.split,
            pair_index=case.pair_index,
            seed=case.seed,
            left_side=left_side,
            outcome=outcome,
            left_objective_completed=left_score > initial_scores[left_side],
            right_objective_completed=right_score > initial_scores[right_side],
            left_score=left_score,
            right_score=right_score,
            decisions=decisions,
            terminated=terminated,
            truncated=truncated,
            peer_tic_lag_max=peer_tic_lag_max,
            protocol_inconsistent=False,
            action_tic_inconsistent=action_tic_inconsistent,
            score_event_inconsistent=score_event_inconsistent,
            scenario_hash=reset_info.scenario_hash,
            environment_attempts=1,
        )
    finally:
        environment.close()


def _validation_cases(cases: Sequence[LeagueCase]) -> tuple[LeagueCase, ...]:
    if len(cases) < 10 or any(case.split != "validation" for case in cases):
        raise ValueError("Cross-play requires validation cases")
    for host, opponent in zip(cases[::2], cases[1::2], strict=True):
        if (
            host.pair_index != opponent.pair_index
            or host.seed != opponent.seed
            or (host.learner_side, opponent.learner_side) != ("host", "opponent")
        ):
            raise ValueError("Cross-play validation cases must be paired")
    return tuple(cases[:10])


def evaluate_crossplay(
    policies: Sequence[OpponentSpec],
    cases: Sequence[LeagueCase],
    *,
    episode_runner: EpisodeRunner,
) -> tuple[CrossplayRow, ...]:
    ordered = tuple(sorted(policies, key=lambda policy: policy.opponent_id))
    if not ordered or len({policy.opponent_id for policy in ordered}) != len(ordered):
        raise ValueError("Cross-play policies must be non-empty and unique")
    if len({policy.scenario_hash for policy in ordered}) != 1:
        raise ValueError("Cross-play policy scenario hashes do not match")
    selected_cases = _validation_cases(cases)
    rows: list[CrossplayRow] = []
    for left, right in combinations_with_replacement(ordered, 2):
        for case in selected_cases:
            row = episode_runner(left, right, case)
            expected = (
                left.opponent_id,
                right.opponent_id,
                case.split,
                case.pair_index,
                case.seed,
                case.learner_side,
            )
            actual = (
                row.left_policy,
                row.right_policy,
                row.split,
                row.pair_index,
                row.seed,
                row.left_side,
            )
            if actual != expected or row.scenario_hash != left.scenario_hash:
                raise ValueError("Cross-play runner returned mismatched row identity")
            rows.append(row)
    return tuple(rows)


def _matrix(policy_ids: Sequence[str], value: object) -> dict[str, dict[str, object]]:
    return {
        left: {right: value for right in policy_ids}
        for left in policy_ids
    }


def summarize_payoff_matrix(
    rows: Sequence[CrossplayRow], *, policy_ids: Sequence[str]
) -> dict[str, object]:
    policies = tuple(policy_ids)
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("Payoff matrix policy IDs must be non-empty and unique")
    identities = [
        (row.left_policy, row.right_policy, row.pair_index, row.left_side)
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Cross-play rows contain duplicate identities")
    if any(
        row.protocol_inconsistent
        or row.action_tic_inconsistent
        or row.score_event_inconsistent
        or row.peer_tic_lag_max != 0
        for row in rows
    ):
        raise ValueError("Cross-play rows contain protocol inconsistencies")
    grouped: dict[tuple[str, str], list[CrossplayRow]] = {}
    for row in rows:
        grouped.setdefault((row.left_policy, row.right_policy), []).append(row)
    expected_pairs = set(combinations_with_replacement(sorted(policies), 2))
    if set(grouped) != expected_pairs:
        raise ValueError("Cross-play matrix is not complete")
    for pair_rows in grouped.values():
        pair_indices = {row.pair_index for row in pair_rows}
        if len(pair_rows) != 10 or len(pair_indices) != 5 or any(
            {row.left_side for row in pair_rows if row.pair_index == pair_index}
            != {"host", "opponent"}
            for pair_index in pair_indices
        ):
            raise ValueError("Cross-play matrix is not complete and paired")
        if any(
            not math.isfinite(float(row.score_difference)) for row in pair_rows
        ):
            raise ValueError("Cross-play matrix contains non-finite values")

    games = _matrix(policies, 0)
    wins = _matrix(policies, 0.0)
    draws = _matrix(policies, 0.0)
    objectives = _matrix(policies, 0.0)
    scores = _matrix(policies, 0.0)
    win_intervals: dict[str, dict[str, dict[str, float]]] = {
        left: {} for left in policies
    }
    draw_intervals: dict[str, dict[str, dict[str, float]]] = {
        left: {} for left in policies
    }
    objective_intervals: dict[str, dict[str, dict[str, float]]] = {
        left: {} for left in policies
    }
    for left in policies:
        for right in policies:
            pair = tuple(sorted((left, right)))
            pair_rows = grouped[pair]
            forward = left == pair[0]
            trials = len(pair_rows)
            games[left][right] = trials
            win_count = sum(
                row.outcome == ("win" if forward else "loss") for row in pair_rows
            )
            draw_count = sum(row.outcome == "draw" for row in pair_rows)
            objective_count = sum(
                row.left_objective_completed
                if forward
                else row.right_objective_completed
                for row in pair_rows
            )
            wins[left][right] = win_count / trials
            draws[left][right] = draw_count / trials
            objectives[left][right] = objective_count / trials
            for target, successes in (
                (win_intervals, win_count),
                (draw_intervals, draw_count),
                (objective_intervals, objective_count),
            ):
                lower, upper = wilson_interval(successes, trials)
                target[left][right] = {"lower": lower, "upper": upper}
            direction = 1 if forward else -1
            scores[left][right] = sum(
                direction * row.score_difference for row in pair_rows
            ) / trials
    return {
        "policy_ids": list(policies),
        "executed_rows": len(rows),
        "games": games,
        "win_rate": wins,
        "draw_rate": draws,
        "objective_rate": objectives,
        "score_difference": scores,
        "win_rate_ci95": win_intervals,
        "draw_rate_ci95": draw_intervals,
        "objective_rate_ci95": objective_intervals,
    }


def write_crossplay_csv_atomic(rows: Sequence[CrossplayRow], path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(CrossplayRow.__dataclass_fields__) + ["score_difference"]
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            writer = csv.DictWriter(
                temporary, fieldnames=fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(row.to_dict() for row in rows)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
