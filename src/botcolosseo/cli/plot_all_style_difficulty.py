from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

_DIFFICULTIES = ("easy", "normal", "hard")
_POLICIES = (
    ("strong_base", "Strong Base", "#334155", "o"),
    ("aggressive", "Aggressive", "#dc2626", "s"),
    ("defensive", "Defensive", "#2563eb", "^"),
    ("explorer", "Explorer", "#059669", "D"),
)


def render_all_style_difficulty_chart(summary_path: Path, output: Path) -> Path:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        not isinstance(summary, dict)
        or summary.get("stage") != "m5-all-style-difficulty"
        or summary.get("split") != "validation"
        or summary.get("passed") is not True
        or summary.get("complete") is not True
        or summary.get("episodes") != 1800
        or summary.get("test_cases_accessed") is not False
    ):
        raise ValueError(
            "All-style chart requires passing 1,800-episode evidence"
        )
    gates = summary.get("gates")
    cells = summary.get("cells")
    if (
        not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
        or not isinstance(cells, Mapping)
    ):
        raise ValueError("All-style chart gates or cells are invalid")

    performance: dict[str, list[float]] = {}
    for policy, _, _, _ in _POLICIES:
        by_difficulty = cells.get(policy)
        if not isinstance(by_difficulty, Mapping):
            raise ValueError("All-style chart policy cells are missing")
        performance[policy] = []
        for difficulty in _DIFFICULTIES:
            cell = by_difficulty.get(difficulty)
            if not isinstance(cell, Mapping) or cell.get("episodes") != 100:
                raise ValueError("All-style chart cell evidence is incomplete")
            performance[policy].append(_rate(cell.get("performance")))

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    figure = Figure(figsize=(9.2, 5.0), dpi=150, facecolor="#f8fafc")
    FigureCanvasAgg(figure)
    try:
        axis = figure.add_subplot(1, 1, 1)
        x = range(len(_DIFFICULTIES))
        for policy, label, color, marker in _POLICIES:
            axis.plot(
                x,
                performance[policy],
                marker=marker,
                linewidth=2.2,
                markersize=7,
                label=label,
                color=color,
            )
        axis.set_xticks(tuple(x), ("Easy", "Normal", "Hard (native)"))
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Performance = 0.5 × (outcome + objective)")
        axis.set_title(
            "Style × Difficulty | 1,800 paired validation episodes"
        )
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="lower right", ncols=2)
        axis.text(
            0.01,
            0.02,
            "All three native style gates pass at every tier",
            transform=axis.transAxes,
            color="#475569",
            fontsize=9,
        )
        figure.tight_layout()
        figure.savefig(temporary, facecolor=figure.get_facecolor())
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        figure.clear()
    return output


def _rate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("All-style chart rate is invalid")
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError("All-style chart rate must be in [0, 1]")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render M5 all-style difficulty evidence"
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = render_all_style_difficulty_chart(args.summary, args.output)
    print(output)
    return 0
