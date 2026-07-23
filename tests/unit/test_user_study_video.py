from __future__ import annotations

from botcolosseo.evaluation.user_study_video import rank_user_study_candidates


def _common(policy: str, pair_index: int) -> dict[str, object]:
    return {
        "policy": policy,
        "split": "validation",
        "opponent": "fixed_route",
        "pair_index": pair_index,
        "learner_side": "host",
        "objective_completed": True,
        "protocol_inconsistent": False,
        "environment_attempts": 1,
        "terminated": True,
        "truncated": False,
    }


def test_rank_user_study_candidates_uses_style_specific_signals() -> None:
    aggressive = [
        {
            **_common("aggressive", index),
            "valid_hits": hits,
            "engagement_initiations": index,
            "attack_decisions": 10,
        }
        for index, hits in ((1, 2), (2, 8), (3, 5))
    ]
    defensive = [
        {
            **_common("defensive", index),
            "recoveries": recovery,
            "successful_escapes": escapes,
            "low_health_opportunities": low_health,
            "risk_decisions": 20,
        }
        for index, recovery, escapes, low_health in (
            (1, 0, 2, 30),
            (2, 1, 0, 40),
            (3, 1, 2, 20),
        )
    ]
    explorer = [
        {
            **_common("explorer", index),
            "upper_completions": upper,
            "lower_completions": lower,
            "flank_completions": 0,
            "route_entropy": entropy,
            "completed_routes": upper + lower,
            "unique_regions": 6,
        }
        for index, upper, lower, entropy in (
            (1, 2, 0, 0.0),
            (2, 1, 1, 0.6),
            (3, 2, 1, 0.5),
        )
    ]

    result = rank_user_study_candidates(
        aggressive_records=aggressive,
        defensive_records=defensive,
        explorer_records=explorer,
        limit=2,
    )

    assert [row["case_id"] for row in result["aggressive"]] == [
        "fixed_route:2:host",
        "fixed_route:3:host",
    ]
    assert [row["case_id"] for row in result["defensive"]] == [
        "fixed_route:2:host",
        "fixed_route:1:host",
    ]
    assert [row["case_id"] for row in result["explorer"]] == [
        "fixed_route:2:host",
        "fixed_route:3:host",
    ]
