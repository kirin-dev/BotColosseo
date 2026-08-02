from __future__ import annotations

from pathlib import Path

import pytest

from botcolosseo.data.extraction_demonstrations import ExtractionCase
from botcolosseo.training.extraction_pfsp import (
    ExtractionHistoricalOpponent,
    ExtractionPFSPSchedule,
    LayoutCurriculumStage,
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
    assert all(
        left.opponent_id == right.opponent_id
        for left, right in zip(assignments[::2], assignments[1::2], strict=True)
    )

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


def test_layout_curriculum_is_stratified_and_step_driven() -> None:
    randomized = tuple(
        ExtractionCase(
            "train",
            seed,
            side,
            ("strong", "aggressive", "defensive", "explorer")[seed % 4],
            "randomized",
        )
        for seed in range(128)
        for side in ("host", "opponent")
    )
    schedule = ExtractionPFSPSchedule(
        randomized,
        shaping_decay_steps=800_000,
        master_seed=7,
        history_probability=0,
        layout_curriculum=(
            LayoutCurriculumStage(0, 16),
            LayoutCurriculumStage(100_000, 32),
            LayoutCurriculumStage(300_000, 128),
        ),
    )

    assert schedule.layout_variant_limit(99_999) == 16
    assert schedule.layout_variant_limit(100_000) == 32
    assert schedule.layout_variant_limit(300_000) == 128
    assignments = [schedule.assignment(0, index) for index in range(32)]
    assert {item.case.opponent_style for item in assignments} == {
        "strong",
        "aggressive",
        "defensive",
        "explorer",
    }
    assert all(
        left.case.seed == right.case.seed
        and left.case.learner_side == "host"
        and right.case.learner_side == "opponent"
        for left, right in zip(assignments[::2], assignments[1::2], strict=True)
    )
    assert max(item.case.seed % 128 for item in assignments) < 16
    assert assignments == [schedule.assignment(0, index) for index in range(32)]
