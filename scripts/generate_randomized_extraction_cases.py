from __future__ import annotations

import json
from pathlib import Path


def write(path: Path, *, split: str, seeds: range) -> None:
    styles = ("strong", "aggressive", "defensive", "explorer", "random_legal")
    cases = []
    for index, seed in enumerate(seeds):
        style = styles[index % len(styles)]
        if split != "train" and style == "random_legal":
            style = "strong"
        for side in ("host", "opponent"):
            cases.append(
                {
                    "split": split,
                    "seed": seed,
                    "learner_side": side,
                    "opponent_style": style,
                    "layout_id": "randomized",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"schema_version": 1, "split": split, "paired_side_swaps": True, "cases": cases},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_evaluation(path: Path, *, layout_id: str, seeds: range) -> None:
    styles = ("strong", "aggressive", "defensive", "explorer")
    cases = [
        {
            "split": "validation",
            "seed": seed,
            "learner_side": side,
            "opponent_style": styles[index % 4],
            "layout_id": layout_id,
        }
        for index, seed in enumerate(seeds)
        for side in ("host", "opponent")
    ]
    path.write_text(
        json.dumps(
            {"schema_version": 1, "split": "validation", "paired_side_swaps": True, "cases": cases},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    write(
        Path("configs/extraction/randomized/train.json"), split="train", seeds=range(51200, 51328)
    )
    write(
        Path("configs/extraction/randomized/validation.json"),
        split="validation",
        seeds=range(61000, 61032),
    )
    for name, layout in (
        ("base", "base"),
        ("heldout", "heldout-a"),
        ("unseen-random", "randomized"),
    ):
        write_evaluation(
            Path(f"configs/extraction/randomized/evaluation-{name}.json"),
            layout_id=layout,
            seeds=range(62000, 62016),
        )
    write_evaluation(
        Path("configs/extraction/randomized/aligned-v2/validation-120.json"),
        layout_id="randomized",
        seeds=range(63000, 63060),
    )
