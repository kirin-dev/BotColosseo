from __future__ import annotations

import json
from pathlib import Path

from botcolosseo.training.extraction_run_log import (
    reconcile_extraction_metrics,
)


def write(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )


def test_extraction_metrics_reconcile_to_checkpoint_boundary(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write(
        path,
        [
            {
                "kind": "rollout",
                "environment_steps": 32,
                "event_counts": {"learner:hit": 2},
                "reward_components": {"progress": 0.5},
            },
            {"kind": "episode", "seed": 1},
            {"kind": "train", "environment_steps": 32},
            {
                "kind": "rollout",
                "environment_steps": 64,
                "event_counts": {"learner:hit": 3},
                "reward_components": {"progress": 0.7},
            },
            {"kind": "train", "environment_steps": 64},
        ],
    )

    history = reconcile_extraction_metrics(
        path,
        committed_environment_steps=32,
    )

    assert history.environment_steps == 32
    assert history.episodes == 1
    assert history.event_counts == {"learner:hit": 2}
    assert history.reward_components == {"progress": 0.5}
    assert '"environment_steps": 64' not in path.read_text(encoding="utf-8")
