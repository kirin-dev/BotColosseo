from __future__ import annotations

import json
from pathlib import Path

import pytest

from botcolosseo.cli.plot_difficulty import render_difficulty_chart


def _summary(path: Path, *, passed: bool = True) -> None:
    cells = {}
    for policy, offset in (("strong_base", 0.0), ("aggressive", 0.01)):
        cells[policy] = {
            difficulty: {
                "episodes": 100,
                "performance": value + offset,
                "objective_rate": min(value + 0.05, 1.0),
            }
            for difficulty, value in (
                ("easy", 0.82),
                ("normal", 0.88),
                ("hard", 0.95),
            )
        }
    path.write_text(
        json.dumps(
            {
                "stage": "m5-difficulty",
                "split": "validation",
                "passed": passed,
                "complete": True,
                "episodes": 600,
                "test_cases_accessed": False,
                "gates": {"monotonic": passed, "style": passed},
                "cells": cells,
            }
        ),
        encoding="utf-8",
    )


def test_difficulty_chart_is_a_nonempty_png(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    _summary(summary)

    output = render_difficulty_chart(summary, tmp_path / "difficulty.png")

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 10_000


def test_difficulty_chart_rejects_failed_evidence(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    _summary(summary, passed=False)

    with pytest.raises(ValueError, match="passing"):
        render_difficulty_chart(summary, tmp_path / "difficulty.png")
