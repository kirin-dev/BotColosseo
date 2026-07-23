from __future__ import annotations

import json
from pathlib import Path

import pytest

from botcolosseo.cli.plot_all_style_difficulty import (
    render_all_style_difficulty_chart,
)


def _summary(path: Path, *, passed: bool = True) -> None:
    cells = {
        policy: {
            difficulty: {
                "episodes": 100,
                "performance": value + offset,
                "objective_rate": min(value + offset + 0.05, 1.0),
            }
            for difficulty, value in (
                ("easy", 0.62),
                ("normal", 0.75),
                ("hard", 0.88),
            )
        }
        for policy, offset in (
            ("strong_base", 0.00),
            ("aggressive", 0.01),
            ("defensive", -0.01),
            ("explorer", 0.02),
        )
    }
    path.write_text(
        json.dumps(
            {
                "stage": "m5-all-style-difficulty",
                "split": "validation",
                "passed": passed,
                "complete": True,
                "episodes": 1800,
                "test_cases_accessed": False,
                "gates": {"all_blocks": passed, "hashes": passed},
                "cells": cells,
            }
        ),
        encoding="utf-8",
    )


def test_all_style_difficulty_chart_is_a_nonempty_png(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    _summary(summary)

    output = render_all_style_difficulty_chart(
        summary,
        tmp_path / "all-style.png",
    )

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 20_000


def test_all_style_difficulty_chart_rejects_failed_evidence(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.json"
    _summary(summary, passed=False)

    with pytest.raises(ValueError, match="passing"):
        render_all_style_difficulty_chart(summary, tmp_path / "all-style.png")
