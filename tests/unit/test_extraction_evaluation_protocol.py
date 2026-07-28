from __future__ import annotations

from pathlib import Path

from botcolosseo.evaluation.extraction_protocol import (
    load_extraction_evaluation_protocol,
    protocol_manifest,
)


def test_v3_evaluation_protocol_freezes_budgets_and_side_swaps() -> None:
    root = Path(__file__).resolve().parents[2]
    protocol = load_extraction_evaluation_protocol(
        root / "configs/extraction/evaluation.yaml"
    )

    assert len(protocol.cases("validation")) == 240
    assert len(protocol.cases("heldout")) == 120
    assert len(protocol.cases("solo")) == 40
    assert set(protocol.splits) == {"validation", "heldout", "solo"}
    assert {case.opponent_style for case in protocol.cases("solo")} == {"idle"}
    first, second = protocol.cases("validation")[:2]
    assert first.seed == second.seed
    assert (first.learner_side, second.learner_side) == ("host", "opponent")
    assert protocol_manifest(protocol)["test_cases_accessed"] is False
