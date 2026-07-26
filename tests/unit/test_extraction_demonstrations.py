from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import botcolosseo.data.extraction_demonstrations as demonstrations
from botcolosseo.agents.extraction_teachers import ExtractionStyle
from botcolosseo.data.extraction_demonstrations import (
    ExtractionDemonstrationBuffer,
    extraction_scalars,
    load_extraction_cases,
    load_extraction_shard,
    write_extraction_shard,
)
from botcolosseo.envs.extraction_types import ExtractionActorObservation
from botcolosseo.envs.ipc import WorkerTimeout


def observation() -> ExtractionActorObservation:
    return ExtractionActorObservation(
        frame=np.zeros((84, 84), dtype=np.uint8),
        health=80,
        ammo=20,
        carried_value=85,
        free_slots=0,
        minimum_slot_value=10,
        banked_value=0,
        extraction_open=True,
        extraction_progress=0.5,
        remaining_time=30,
        previous_action=1,
    )


def test_extraction_scalars_are_public_and_normalized() -> None:
    scalars = extraction_scalars(observation())

    assert scalars.shape == (9,)
    assert scalars.dtype == np.float32
    assert scalars.tolist() == pytest.approx(
        [0.8, 0.5, 85 / 150, 0, 0.2, 0, 1, 0.5, 0.4]
    )


def test_extraction_shard_round_trip(tmp_path: Path) -> None:
    buffer = ExtractionDemonstrationBuffer()
    buffer.append(
        observation(),
        teacher_action=10,
        episode_start=True,
        style=ExtractionStyle.STRONG,
        train_seed=7,
    )

    path = write_extraction_shard(buffer.arrays(), tmp_path / "shard.npz")
    loaded = load_extraction_shard(path)

    assert loaded["frame"].shape == (1, 84, 84)
    assert loaded["scalars"].shape == (1, 9)
    assert loaded["teacher_action"].tolist() == [10]
    assert loaded["style_id"].tolist() == [0]


def test_generation_case_loader_rejects_test_access(tmp_path: Path) -> None:
    path = tmp_path / "test.json"
    path.write_text(
        json.dumps(
            {
                "split": "test",
                "cases": [
                    {
                        "split": "test",
                        "seed": 1,
                        "learner_side": "host",
                        "opponent_style": "strong",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot access test"):
        load_extraction_cases(path, expected_split="test")


def test_extraction_episode_generation_retries_transient_startup_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = ExtractionDemonstrationBuffer()
    calls = 0

    def fake_collect(**kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        if calls < 3:
            raise WorkerTimeout("port collision")
        return expected, {}

    monkeypatch.setattr(demonstrations, "_collect_episode_once", fake_collect)
    case = demonstrations.ExtractionCase(
        split="train",
        seed=1,
        learner_side="host",
        opponent_style="strong",
    )

    result, events = demonstrations._collect_episode(
        root=tmp_path,
        case=case,
        style=ExtractionStyle.STRONG,
        max_decisions=10,
    )

    assert result is expected
    assert events == {}
    assert calls == 3
