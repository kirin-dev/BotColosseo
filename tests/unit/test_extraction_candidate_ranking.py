from __future__ import annotations

from botcolosseo.cli.rank_extraction_candidates import _strong_score


def test_strong_candidate_ranking_prioritizes_gate_coverage() -> None:
    passing = {
        "metrics": {
            "prevent_opponent_extraction_rate": 0.91,
            "win_rate": 0.71,
            "extraction_rate": 0.76,
            "mean_extracted_value_advantage": 1,
            "protocol_inconsistencies": 0,
            "by_opponent": {
                name: {"win_rate": 0.56}
                for name in ("strong", "aggressive", "defensive", "explorer")
            },
        }
    }
    high_win_but_incomplete = {
        "metrics": {
            **passing["metrics"],
            "win_rate": 0.9,
            "by_opponent": {
                **passing["metrics"]["by_opponent"],
                "strong": {"win_rate": 0.5},
            },
        }
    }

    assert _strong_score(passing) > _strong_score(high_win_but_incomplete)
