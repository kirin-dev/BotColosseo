from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from botcolosseo.envs.actions import MacroAction
from botcolosseo.evaluation.m3 import (
    EXPECTED_CORE_LOCATIONS,
    M3EpisodeRecord,
    NoOpponentController,
    evaluate_m3_records,
    expected_m3_episode_count,
)
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS

HISTORICAL = tuple(f"historical-{index:02d}" for index in range(8))


def test_evaluation_config_freezes_counts_thresholds_and_bootstrap() -> None:
    config = yaml.safe_load(
        Path("configs/m3/evaluation.yaml").read_text(encoding="utf-8")
    )

    assert config["script_pairs_per_opponent"] == 50
    assert config["no_opponent_pairs"] == 50
    assert config["heldout_pairs"] == 50
    assert config["historical_pairs_per_policy"] == 20
    assert (config["pool_size_min"], config["pool_size_max"]) == (8, 12)
    assert config["bootstrap"] == {
        "seed": 20260721,
        "samples": 10_000,
        "confidence": 0.95,
    }
    assert config["thresholds"] == {
        "script_average_win_rate": 0.70,
        "script_per_opponent_win_rate": 0.55,
        "no_opponent_full_objective_rate": 0.90,
        "heldout_full_objective_rate": 0.80,
        "paired_score_lcb": 0.0,
    }


def _record(
    *,
    policy: str,
    category: str,
    split: str,
    opponent: str,
    pair_index: int,
    side: str,
    win: bool,
    objective: bool,
    score_difference: int,
    core: tuple[float, float] = (0.0, 0.0),
) -> M3EpisodeRecord:
    return M3EpisodeRecord(
        policy=policy,
        category=category,
        split=split,
        opponent=opponent,
        pair_index=pair_index,
        seed=10_000 + pair_index,
        learner_side=side,
        outcome="win" if win else "loss",
        objective_completed=objective,
        goal_reached=objective,
        pickup_completed=objective,
        return_completed=objective,
        valid_hit=win,
        disengage_success=objective,
        learner_score=max(score_difference, 0),
        opponent_score=max(-score_difference, 0),
        actual_core_x=core[0],
        actual_core_y=core[1],
        decisions=20,
        terminated=True,
        truncated=False,
        peer_tic_lag_max=0,
        protocol_inconsistent=False,
        action_tic_inconsistent=False,
        score_event_inconsistent=False,
        fairness_schema_inconsistent=False,
        scenario_hash="scenario",
        environment_attempts=1,
    )


def _passing_records() -> list[M3EpisodeRecord]:
    rows: list[M3EpisodeRecord] = []
    for opponent_index, opponent in enumerate(DUEL_OPPONENTS):
        for pair in range(50):
            win = pair % 10 < 7
            objective = pair % 10 < 9
            for side in ("host", "opponent"):
                rows.append(
                    _record(
                        policy="strong_base",
                        category="script",
                        split="test",
                        opponent=opponent,
                        pair_index=opponent_index * 100 + pair,
                        side=side,
                        win=win,
                        objective=objective,
                        score_difference=2 if win else -1,
                    )
                )
    for pair in range(50):
        objective = pair % 10 < 9
        for side in ("host", "opponent"):
            rows.append(
                _record(
                    policy="strong_base",
                    category="no_opponent",
                    split="test",
                    opponent="no_opponent",
                    pair_index=1_000 + pair,
                    side=side,
                    win=objective,
                    objective=objective,
                    score_difference=1 if objective else -1,
                )
            )
    for pair in range(50):
        objective = pair % 10 < 8
        opponent = DUEL_OPPONENTS[pair % len(DUEL_OPPONENTS)]
        core = EXPECTED_CORE_LOCATIONS[pair % len(EXPECTED_CORE_LOCATIONS)]
        for side in ("host", "opponent"):
            rows.append(
                _record(
                    policy="strong_base",
                    category="heldout",
                    split="heldout",
                    opponent=opponent,
                    pair_index=2_000 + pair,
                    side=side,
                    win=objective,
                    objective=objective,
                    score_difference=1 if objective else -1,
                    core=core,
                )
            )
    for opponent_index, opponent in enumerate(HISTORICAL):
        for pair in range(20):
            for policy in ("strong_base", "m2_baseline"):
                win = pair < (13 if policy == "strong_base" else 10)
                difference = (2 if win else -1) if policy == "strong_base" else (1 if win else -2)
                for side in ("host", "opponent"):
                    rows.append(
                        _record(
                            policy=policy,
                            category="historical",
                            split="test",
                            opponent=opponent,
                            pair_index=3_000 + opponent_index * 100 + pair,
                            side=side,
                            win=win,
                            objective=win,
                            score_difference=difference,
                        )
                    )
    return rows


def _summary(rows: list[M3EpisodeRecord]):
    return evaluate_m3_records(
        rows,
        historical_policy_ids=HISTORICAL,
        expected_scenario_hash="scenario",
    )


def test_frozen_episode_count_and_complete_report_schema() -> None:
    summary = _summary(_passing_records())
    payload = summary.to_dict()

    assert expected_m3_episode_count(8) == 1_340
    assert summary.episodes == summary.expected_episodes == 1_340
    assert summary.complete is True
    assert summary.passed is True
    assert set(payload) == {
        "schema_version",
        "official",
        "complete",
        "passed",
        "episodes",
        "expected_episodes",
        "pool_size",
        "protocol_inconsistencies",
        "protocol_counts",
        "artifact_inconsistencies",
        "environment_retries",
        "paired_historical_score_difference",
        "gates",
        "categories",
        "historical_by_policy",
        "heldout_core_strata",
    }
    assert summary.categories["script"].wins.rate == pytest.approx(0.70)
    assert summary.categories["no_opponent"].objectives.rate == pytest.approx(0.90)
    assert summary.categories["heldout"].objectives.rate == pytest.approx(0.80)
    assert len(summary.heldout_core_strata) == 3
    assert summary.paired_historical_score_difference.samples == 10_000


def test_each_category_threshold_is_conjunctive() -> None:
    rows = _passing_records()
    script_failure = [
        replace(row, outcome="loss", learner_score=0, opponent_score=1)
        if row.category == "script"
        and row.opponent == "aggressive_script"
        and row.pair_index % 100 in (0, 1, 2, 3, 4, 5, 6, 10)
        else row
        for row in rows
    ]
    assert _summary(script_failure).gates["script_per_opponent_win_rate"] is False

    no_opponent = list(rows)
    index = next(
        index
        for index, row in enumerate(no_opponent)
        if row.category == "no_opponent" and row.objective_completed
    )
    no_opponent[index] = replace(no_opponent[index], objective_completed=False)
    assert _summary(no_opponent).gates["no_opponent_full_objective_rate"] is False

    heldout = list(rows)
    index = next(
        index
        for index, row in enumerate(heldout)
        if row.category == "heldout" and row.objective_completed
    )
    heldout[index] = replace(heldout[index], objective_completed=False)
    assert _summary(heldout).gates["heldout_full_objective_rate"] is False


def test_historical_worst_case_and_paired_score_lcb_are_separate_gates() -> None:
    rows = _passing_records()
    worst_case_failure = [
        replace(row, outcome="loss", learner_score=0, opponent_score=1)
        if row.category == "historical"
        and row.policy == "strong_base"
        and row.opponent == HISTORICAL[0]
        and row.pair_index % 100 in (10, 11, 12)
        else row
        for row in rows
    ]
    assert _summary(worst_case_failure).gates["historical_worst_case_improved"] is False

    score_failure = []
    for row in rows:
        if row.category == "historical":
            if row.policy == "strong_base":
                scores = (1, 0) if row.outcome == "win" else (0, 100)
            else:
                scores = (100, 0) if row.outcome == "win" else (0, 1)
            row = replace(row, learner_score=scores[0], opponent_score=scores[1])
        score_failure.append(row)
    failed = _summary(score_failure)
    assert failed.gates["historical_worst_case_improved"] is True
    assert failed.gates["paired_score_lcb"] is False


def test_missing_dirty_unpaired_or_missing_core_strata_fail_closed() -> None:
    rows = _passing_records()
    assert _summary(rows[:-1]).complete is False
    dirty = list(rows)
    dirty[0] = replace(dirty[0], protocol_inconsistent=True)
    dirty_summary = _summary(dirty)
    assert dirty_summary.protocol_inconsistencies == 1
    assert dirty_summary.passed is False
    unpaired = list(rows)
    unpaired[0] = replace(unpaired[0], learner_side="opponent")
    assert _summary(unpaired).complete is False
    baseline_seed_mismatch = list(rows)
    baseline_pair = next(
        row.pair_index
        for row in baseline_seed_mismatch
        if row.category == "historical" and row.policy == "m2_baseline"
    )
    baseline_seed_mismatch = [
        replace(row, seed=row.seed + 1)
        if row.category == "historical"
        and row.policy == "m2_baseline"
        and row.pair_index == baseline_pair
        else row
        for row in baseline_seed_mismatch
    ]
    assert _summary(baseline_seed_mismatch).complete is False
    one_core = [
        replace(row, actual_core_x=0.0, actual_core_y=0.0)
        if row.category == "heldout"
        else row
        for row in rows
    ]
    assert _summary(one_core).gates["heldout_core_strata_complete"] is False


def test_no_opponent_controller_is_literal_idle_and_stateless() -> None:
    controller = NoOpponentController()
    controller.reset(seed=7)

    assert controller.name == "no_opponent"
    assert controller.act(object(), object()) is MacroAction.IDLE
