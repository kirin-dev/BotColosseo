from __future__ import annotations

from pathlib import Path

import pytest

from botcolosseo.data.extraction_demonstrations import ExtractionCase
from botcolosseo.training.extraction_pfsp import (
    ExtractionHistoricalOpponent,
    ExtractionPFSPSchedule,
)


def cases() -> tuple[ExtractionCase, ...]:
    return (
        ExtractionCase("train", 1, "host", "strong"),
        ExtractionCase("train", 1, "opponent", "strong"),
        ExtractionCase("train", 2, "host", "aggressive"),
        ExtractionCase("train", 2, "opponent", "aggressive"),
    )


def opponent(tmp_path: Path, index: int) -> ExtractionHistoricalOpponent:
    checkpoint = tmp_path / f"candidate-{index}.pt"
    checkpoint.write_bytes(str(index).encode())
    return ExtractionHistoricalOpponent(
        opponent_id=f"strong-{index}",
        checkpoint=checkpoint,
        checkpoint_sha256=f"{index:064x}",
        environment_steps=index,
    )


def test_extraction_pfsp_is_deterministic_and_updates_payoffs(tmp_path: Path) -> None:
    schedule = ExtractionPFSPSchedule(
        cases(),
        shaping_decay_steps=100,
        master_seed=7,
        history_probability=1,
    )
    first = opponent(tmp_path, 1)
    second = opponent(tmp_path, 2)
    assert schedule.add(first)
    assert schedule.add(second)
    assert not schedule.add(first)

    assignments = [schedule.assignment(0, index) for index in range(20)]
    repeated = [schedule.assignment(0, index) for index in range(20)]
    assert assignments == repeated
    assert all(item.opponent_kind == "checkpoint" for item in assignments)
    assert {item.case.learner_side for item in assignments} == {"host", "opponent"}

    schedule.record_result(assignments[0], won=True, draw=False)
    schedule.record_result(assignments[1], won=False, draw=False)
    assert schedule.win_rates[assignments[0].opponent_id] in {0.0, 0.5, 1.0}
    assert schedule.shaping_scale(50) == pytest.approx(0.5)
    restored = ExtractionPFSPSchedule(
        cases(),
        shaping_decay_steps=100,
        master_seed=7,
        history_probability=1,
    )
    restored.add(first)
    restored.add(second)
    restored.load_state_dict(schedule.state_dict())
    assert restored.win_rates == schedule.win_rates


def test_extraction_pfsp_rejects_duplicate_checkpoint(tmp_path: Path) -> None:
    schedule = ExtractionPFSPSchedule(
        cases(),
        shaping_decay_steps=100,
        master_seed=7,
    )
    first = opponent(tmp_path, 1)
    schedule.add(first)
    duplicate = ExtractionHistoricalOpponent(
        opponent_id="other",
        checkpoint=first.checkpoint,
        checkpoint_sha256=first.checkpoint_sha256,
        environment_steps=2,
    )
    with pytest.raises(ValueError, match="duplicated"):
        schedule.add(duplicate)
