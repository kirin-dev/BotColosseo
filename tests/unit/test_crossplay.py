import dataclasses
from pathlib import Path

import numpy as np
import pytest

from botcolosseo.agents.league_opponents import OpponentSpec
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_protocol import DuelEvent, DuelEventType
from botcolosseo.envs.duel_types import DuelActorObservation, DuelPrivilegedState, DuelStep
from botcolosseo.envs.synchronous_duel import DuelObservations, DuelResetInfo
from botcolosseo.evaluation.crossplay import (
    CrossplayRow,
    evaluate_crossplay,
    run_crossplay_episode,
    summarize_payoff_matrix,
    write_crossplay_csv_atomic,
)
from botcolosseo.scenarios.league_splits import generate_league_splits
from botcolosseo.scenarios.regions import RegionGraph


def _spec(policy_id: str) -> OpponentSpec:
    return OpponentSpec(
        opponent_id=policy_id,
        kind="script",
        checkpoint=None,
        checkpoint_sha256=None,
        scenario_hash="scenario",
        selection_evidence=f"builtin:{policy_id}",
    )


def _runner(left: OpponentSpec, right: OpponentSpec, case) -> CrossplayRow:
    if left.opponent_id == right.opponent_id:
        left_score = right_score = 1
    else:
        left_score, right_score = 2, 0
    difference = left_score - right_score
    return CrossplayRow(
        left_policy=left.opponent_id,
        right_policy=right.opponent_id,
        split=case.split,
        pair_index=case.pair_index,
        seed=case.seed,
        left_side=case.learner_side,
        outcome="win" if difference > 0 else "draw" if difference == 0 else "loss",
        left_objective_completed=left_score > 0,
        right_objective_completed=right_score > 0,
        left_score=left_score,
        right_score=right_score,
        decisions=20,
        terminated=True,
        truncated=False,
        peer_tic_lag_max=0,
        protocol_inconsistent=False,
        action_tic_inconsistent=False,
        score_event_inconsistent=False,
        scenario_hash="scenario",
        environment_attempts=1,
    )


def test_crossplay_executes_each_unordered_pair_once_and_derives_ordered_matrix() -> None:
    cases = generate_league_splits()["validation"]

    rows = evaluate_crossplay(
        (_spec("policy-b"), _spec("policy-a")),
        cases,
        episode_runner=_runner,
    )
    summary = summarize_payoff_matrix(rows, policy_ids=("policy-a", "policy-b"))

    assert len(rows) == 5 * 2 * 3
    assert [(row.left_policy, row.right_policy) for row in rows[:10]] == [
        ("policy-a", "policy-a")
    ] * 10
    assert summary["games"]["policy-a"]["policy-b"] == 10
    assert summary["games"]["policy-b"]["policy-a"] == 10
    assert summary["win_rate"]["policy-a"]["policy-b"] == 1.0
    assert summary["win_rate"]["policy-b"]["policy-a"] == 0.0
    assert summary["score_difference"]["policy-a"]["policy-b"] == 2.0
    assert summary["score_difference"]["policy-b"]["policy-a"] == -2.0
    assert summary["win_rate_ci95"]["policy-a"]["policy-b"]["lower"] > 0.72
    assert summary["win_rate_ci95"]["policy-a"]["policy-b"]["upper"] == 1.0
    assert summary["objective_rate_ci95"]["policy-b"]["policy-a"][
        "lower"
    ] == pytest.approx(0.0, abs=1e-12)
    assert summary["draw_rate_ci95"]["policy-a"]["policy-a"]["upper"] == 1.0


def test_crossplay_uses_first_five_validation_pairs_with_both_sides() -> None:
    cases = generate_league_splits()["validation"]
    rows = evaluate_crossplay((_spec("policy-a"),), cases, episode_runner=_runner)

    assert len(rows) == 10
    assert [row.pair_index for row in rows] == [case.pair_index for case in cases[:10]]
    assert [row.left_side for row in rows] == ["host", "opponent"] * 5


def test_crossplay_csv_is_atomic_canonical_and_round_trippable(tmp_path: Path) -> None:
    rows = evaluate_crossplay(
        (_spec("policy-a"),),
        generate_league_splits()["validation"],
        episode_runner=_runner,
    )
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"

    write_crossplay_csv_atomic(rows, first)
    write_crossplay_csv_atomic(rows, second)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_text(encoding="utf-8").splitlines()[0].startswith(
        "left_policy,right_policy,split,pair_index"
    )
    assert not list(tmp_path.glob("*.tmp"))


def test_matrix_fails_closed_on_missing_duplicate_or_inconsistent_rows() -> None:
    rows = list(
        evaluate_crossplay(
            (_spec("policy-a"), _spec("policy-b")),
            generate_league_splits()["validation"],
            episode_runner=_runner,
        )
    )
    with pytest.raises(ValueError, match="complete"):
        summarize_payoff_matrix(rows[:-1], policy_ids=("policy-a", "policy-b"))
    with pytest.raises(ValueError, match="duplicate"):
        summarize_payoff_matrix(
            [*rows, rows[-1]], policy_ids=("policy-a", "policy-b")
        )
    rows[0] = dataclasses.replace(rows[0], protocol_inconsistent=True)
    with pytest.raises(ValueError, match="protocol"):
        summarize_payoff_matrix(rows, policy_ids=("policy-a", "policy-b"))


def test_crossplay_rejects_nonvalidation_or_unpaired_cases() -> None:
    splits = generate_league_splits()
    with pytest.raises(ValueError, match="validation"):
        evaluate_crossplay(
            (_spec("policy-a"),), splits["test"], episode_runner=_runner
        )
    with pytest.raises(ValueError, match="paired"):
        evaluate_crossplay(
            (_spec("policy-a"),), splits["validation"][1:], episode_runner=_runner
        )


def _observation(previous_action: int, *, own_score: int = 0, other_score: int = 0):
    return DuelActorObservation(
        frame=np.zeros((84, 84), dtype=np.uint8),
        health=100.0,
        armor=0.0,
        ammo=10.0,
        own_score=own_score,
        opponent_score=other_score,
        has_core=False,
        previous_action=previous_action,
    )


class PublicController:
    def __init__(self, action: MacroAction) -> None:
        self.action = action
        self.previous_actions: list[int] = []
        self.reset_seeds: list[int] = []

    def reset(self, *, seed: int) -> None:
        self.reset_seeds.append(seed)

    def act(self, observation, privileged_state):
        del privileged_state
        self.previous_actions.append(observation.previous_action)
        return self.action


class FakeCrossplayEnv:
    def __init__(self) -> None:
        self.closed = False
        self.scale = 1.0

    def reset(self):
        return (
            DuelObservations(_observation(2), _observation(3)),
            DuelResetInfo(7, 1, 0, 0, 2, "scenario"),
        )

    def set_shaping_scale(self, scale: float) -> None:
        self.scale = scale

    def teacher_state(self) -> DuelPrivilegedState:
        raise AssertionError("public controllers must not resolve privileged state")

    def step(self, host_action, opponent_action) -> DuelStep:
        assert (host_action, opponent_action) == (
            MacroAction.MOVE_FORWARD,
            MacroAction.ATTACK,
        )
        return DuelStep(
            host=_observation(1, own_score=1, other_score=0),
            opponent=_observation(9, own_score=0, other_score=1),
            host_reward=0.0,
            opponent_reward=0.0,
            terminated=True,
            truncated=False,
            events=(DuelEvent(DuelEventType.SCORE, "host", 0, 1, 4),),
            decision_index=1,
            engine_tic=4,
            peer_tic_lag=0,
            pre_action_tics=0,
            action_tics=4,
        )

    def close(self) -> None:
        self.closed = True


def test_crossplay_episode_uses_each_public_perspective_without_eager_privilege() -> None:
    case = generate_league_splits()["validation"][0]
    left = PublicController(MacroAction.MOVE_FORWARD)
    right = PublicController(MacroAction.ATTACK)
    environment = FakeCrossplayEnv()

    row = run_crossplay_episode(
        case,
        left_spec=_spec("policy-a"),
        right_spec=_spec("policy-b"),
        left_controller=left,
        right_controller=right,
        graph=RegionGraph.from_yaml(Path("assets/scenarios/crystal_run/src/regions.yaml")),
        config_path=Path("assets/scenarios/crystal_run/crystal_run.cfg"),
        max_decisions=525,
        environment_factory=lambda selected: environment,
    )

    assert row.outcome == "win"
    assert row.left_objective_completed is True
    assert row.right_objective_completed is False
    assert row.protocol_inconsistent is False
    assert left.previous_actions == [2]
    assert right.previous_actions == [3]
    assert environment.scale == 0.0
    assert environment.closed is True
