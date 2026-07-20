import json
from pathlib import Path

from botcolosseo.cli.train_ppo import (
    _planned_updates,
    _reconcile_metrics,
    _run_hash,
    _validation_cases,
)


def test_planned_updates_handles_partial_final_rollout() -> None:
    assert _planned_updates(
        1000,
        rollout_steps=256,
        sequence_length=16,
        minibatch_sequences=16,
        update_epochs=4,
    ) == 16


def test_resume_discards_metrics_beyond_atomic_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    records = [
        {"kind": "rollout", "environment_steps": 256},
        {"kind": "episode", "episode_index": 0},
        {"kind": "train", "update": 1},
        {"kind": "rollout", "environment_steps": 512},
        {"kind": "episode", "episode_index": 1},
        {"kind": "train", "update": 2},
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in records))

    _reconcile_metrics(path, committed_environment_steps=256)

    kept = [json.loads(line) for line in path.read_text().splitlines()]
    assert kept == records[:3]


def test_run_hash_binds_actual_schedule() -> None:
    first = _run_hash("base", target_steps=1000, rollout_steps=256)

    assert first == _run_hash("base", target_steps=1000, rollout_steps=256)
    assert first != _run_hash("base", target_steps=2000, rollout_steps=256)
    assert first != _run_hash("base", target_steps=1000, rollout_steps=128)


def test_validation_pilot_reads_paired_randomlegal_only() -> None:
    selected = _validation_cases(Path.cwd(), 2)

    assert len(selected) == 4
    assert all(case.split == "validation" for case in selected)
    assert all(case.opponent == "random_legal" for case in selected)
    assert [case.learner_side for case in selected] == [
        "host",
        "opponent",
        "host",
        "opponent",
    ]
