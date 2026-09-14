"""Readable empirical-game figure from a completed, verified role matrix."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from botcolosseo.cli.solve_hierarchical_matrix import solve_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.matrix.read_text())
    solution = solve_report(report)
    arrays = [np.asarray(report[key]) for key in ("A", "B")]
    names = ["Preserve", "Upgrade", "Selective combat", "Response 1", "Response 2"]
    if arrays[0].shape != (5, 5):
        raise ValueError("This showcase figure requires the completed two-round 5x5 game")
    if args.output.exists():
        raise FileExistsError("Preserve existing figure")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    for ax, values, title in zip(
        axes, arrays, ("Host: own extracted value", "Opponent: own extracted value"), strict=True
    ):
        plot = ax.imshow(values, cmap="Blues", vmin=0, vmax=max(a.max() for a in arrays))
        ax.set_xticks(range(5), names, rotation=25, ha="right")
        ax.set_yticks(range(5), names)
        ax.set_xlabel("Opponent strategy")
        ax.set_ylabel("Host strategy")
        ax.set_title(title, fontsize=13, pad=14)
        for i in range(5):
            for j in range(5):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if values[i, j] > 0.55 * plot.norm.vmax else "#18212b",
                )
        fig.colorbar(plot, ax=ax, shrink=0.65, label="Banked value / 150")
    host = names[int(np.argmax(solution["host"]))]
    opponent = names[int(np.argmax(solution["opponent"]))]
    pure = max(solution["host"]) == 1 and max(solution["opponent"]) == 1
    result = f"Pure meta-strategy: {host} / {opponent}" if pure else "Mixed meta-strategy"
    fig.suptitle("Shared executor · Two response rounds\n" + result, fontsize=16)
    fig.supxlabel(
        "48 games per cell · Neutral / Hard · Fixed geometry, randomized loot\n"
        "Estimated restricted game; response training has not established independent gains.",
        fontsize=10,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
