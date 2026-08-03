from botcolosseo.cli.select_randomized_strong_1m import (
    _case_manifest,
    _promotion_gates,
    _rank_key,
)


def report(*, extraction: float, win: float, advantage: float, death: float):
    return {
        "metrics": {
            "episodes": [{}] * 8,
            "extraction_rate": extraction,
            "win_rate": win,
            "mean_extracted_value_advantage": advantage,
            "death_rate": death,
            "protocol_inconsistencies": 0,
        },
        "actor_privilege_violations": 0,
        "test_cases_accessed": False,
    }


def test_selection_manifest_is_paired_and_opponent_balanced() -> None:
    manifest = _case_manifest(episodes=120, layout_id="randomized")
    cases = manifest["cases"]

    assert len(cases) == 120
    assert {case["layout_id"] for case in cases} == {"randomized"}
    assert all(
        left["seed"] == right["seed"]
        and left["opponent_style"] == right["opponent_style"]
        and (left["learner_side"], right["learner_side"])
        == ("host", "opponent")
        for left, right in zip(cases[::2], cases[1::2], strict=True)
    )
    assert {
        style: sum(case["opponent_style"] == style for case in cases)
        for style in ("strong", "aggressive", "defensive", "explorer")
    } == {style: 30 for style in ("strong", "aggressive", "defensive", "explorer")}


def test_selection_ranking_is_lexicographic() -> None:
    higher_extraction = report(extraction=0.8, win=0.1, advantage=-10, death=0.9)
    higher_win = report(extraction=0.7, win=0.6, advantage=0, death=0.2)
    lower_win = report(extraction=0.7, win=0.5, advantage=100, death=0)

    assert _rank_key(higher_extraction) > _rank_key(higher_win)
    assert _rank_key(higher_win) > _rank_key(lower_win)


def test_promotion_gates_compare_against_same_protocol_baseline() -> None:
    baseline = report(extraction=0.65, win=0.45, advantage=0, death=0.30)
    candidate = report(extraction=0.70, win=0.425, advantage=5, death=0.35)
    baseline_base = report(extraction=0.70, win=0, advantage=0, death=0)
    candidate_base = report(extraction=0.65, win=0, advantage=0, death=0)
    baseline_heldout = report(extraction=0.60, win=0, advantage=0, death=0)
    candidate_heldout = report(extraction=0.55, win=0, advantage=0, death=0)

    gates = _promotion_gates(
        candidate,
        baseline_random=baseline,
        candidate_base=candidate_base,
        baseline_base=baseline_base,
        candidate_heldout=candidate_heldout,
        baseline_heldout=baseline_heldout,
    )

    assert all(gates.values())
