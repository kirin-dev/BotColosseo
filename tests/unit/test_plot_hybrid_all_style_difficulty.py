from __future__ import annotations

import json
from pathlib import Path

import pytest

from botcolosseo.cli.plot_hybrid_all_style_difficulty import (
    render_hybrid_all_style_difficulty_chart,
)


def _summary(path: Path, *, passed: bool = True) -> Path:
    payload = {
        "stage": "m5-hybrid-all-style-difficulty",
        "passed": passed,
        "complete": passed,
        "episodes": 1200,
        "test_cases_accessed": False,
        "gates": {"matrix": passed},
        "cells": {
            policy: {
                difficulty: {"episodes": 100, "performance": performance}
                for difficulty, performance in (
                    ("easy", 0.8),
                    ("normal", 0.9),
                    ("hard", 1.0),
                )
            }
            for policy in (
                "strong_base",
                "aggressive",
                "defensive",
                "explorer",
            )
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_hybrid_all_style_chart_renders_passing_matrix(tmp_path: Path) -> None:
    output = render_hybrid_all_style_difficulty_chart(
        _summary(tmp_path / "summary.json"),
        tmp_path / "chart.png",
    )

    assert output.stat().st_size > 10_000


def test_hybrid_all_style_chart_rejects_failed_matrix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="passing"):
        render_hybrid_all_style_difficulty_chart(
            _summary(tmp_path / "summary.json", passed=False),
            tmp_path / "chart.png",
        )
