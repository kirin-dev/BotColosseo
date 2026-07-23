from __future__ import annotations

from copy import deepcopy

from botcolosseo.evaluation.hybrid_all_style_difficulty import (
    evaluate_hybrid_all_style_matrix,
)
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS


def _rows() -> list[dict[str, object]]:
    performance = {"easy": 0.8, "normal": 0.9, "hard": 1.0}
    return [
        {
            "policy": policy,
            "difficulty": difficulty,
            "opponent": opponent,
            "pair_index": pair,
            "learner_side": side,
            "seed": 1000 + opponent_index * 20 + pair * 2 + (side == "opponent"),
            "outcome": "win",
            "objective_completed": True,
            "learner_score": 2,
            "opponent_score": 0,
            "performance": performance[difficulty],
            "terminated": True,
            "truncated": False,
            "peer_tic_lag_max": 0,
            "protocol_inconsistent": False,
            "action_tic_inconsistent": False,
            "score_event_inconsistent": False,
            "scenario_hash": "scenario",
        }
        for policy in ("strong_base", "aggressive", "defensive", "explorer")
        for difficulty in ("easy", "normal", "hard")
        for opponent_index, opponent in enumerate(DUEL_OPPONENTS)
        for pair in range(10)
        for side in ("host", "opponent")
    ]


def test_hybrid_all_style_matrix_accepts_1200_unique_cells() -> None:
    rows = _rows()
    rows[0]["terminated"] = False
    rows[0]["truncated"] = True
    result = evaluate_hybrid_all_style_matrix(
        rows,
        expected_pairs_per_opponent=10,
        expected_scenario_hash="scenario",
        style_source_gates={
            "aggressive": True,
            "defensive": True,
            "explorer": True,
        },
    )

    assert result["passed"] is True
    assert result["episodes"] == 1200
    assert tuple(result["cells"]) == (
        "strong_base",
        "aggressive",
        "defensive",
        "explorer",
    )
    assert all(result["gates"].values())


def test_hybrid_all_style_matrix_rejects_nonmonotonic_policy() -> None:
    rows = _rows()
    for row in rows:
        if row["policy"] == "explorer" and row["difficulty"] == "easy":
            row["performance"] = 1.0
        if row["policy"] == "explorer" and row["difficulty"] == "normal":
            row["performance"] = 0.8

    result = evaluate_hybrid_all_style_matrix(
        rows,
        expected_pairs_per_opponent=10,
        expected_scenario_hash="scenario",
        style_source_gates={
            "aggressive": True,
            "defensive": True,
            "explorer": True,
        },
    )

    assert result["passed"] is False
    assert result["gates"]["aggregate_monotonic"] is False


def test_hybrid_all_style_matrix_rejects_weak_hybrid_retention() -> None:
    rows = deepcopy(_rows())
    for row in rows:
        if row["policy"] == "defensive" and row["difficulty"] == "easy":
            row["performance"] = 0.5

    result = evaluate_hybrid_all_style_matrix(
        rows,
        expected_pairs_per_opponent=10,
        expected_scenario_hash="scenario",
        style_source_gates={
            "aggressive": True,
            "defensive": True,
            "explorer": True,
        },
    )

    assert result["passed"] is False
    assert result["gates"]["hybrid_skill_retention"] is False
