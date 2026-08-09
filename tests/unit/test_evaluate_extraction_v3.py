from __future__ import annotations

from pathlib import Path

from botcolosseo.evaluation.extraction_protocol import (
    balanced_extraction_case_subset,
    load_extraction_evaluation_protocol,
)


def test_balanced_case_subset_keeps_side_swapped_pairs_for_every_opponent() -> None:
    cases = load_extraction_evaluation_protocol(
        Path("configs/extraction/randomized/evaluation.yaml")
    ).cases("validation")

    selected = balanced_extraction_case_subset(cases, pairs_per_opponent=2)

    assert len(selected) == 16
    assert {
        (case.opponent_style, case.learner_side)
        for case in selected
    } == {
        (style, side)
        for style in ("strong", "aggressive", "defensive", "explorer")
        for side in ("host", "opponent")
    }
