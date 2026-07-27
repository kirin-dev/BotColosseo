from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import botcolosseo.data.extraction_demonstrations as demonstrations
from botcolosseo.agents.extraction_teachers import ExtractionStyle
from botcolosseo.data.extraction_demonstrations import (
    ExtractionDemonstrationBuffer,
    generate_extraction_demonstrations,
)
from botcolosseo.envs.extraction_types import ExtractionActorObservation


def episode() -> ExtractionDemonstrationBuffer:
    buffer = ExtractionDemonstrationBuffer()
    observation = ExtractionActorObservation(
        frame=np.zeros((84, 84), dtype=np.uint8),
        health=100,
        ammo=30,
        carried_value=0,
        free_slots=3,
        minimum_slot_value=0,
        banked_value=0,
        extraction_open=False,
        extraction_progress=0,
        remaining_time=75,
        previous_action=0,
    )
    for index in range(2):
        buffer.append(
            observation,
            teacher_action=0,
            episode_start=index == 0,
            style=ExtractionStyle.STRONG,
            train_seed=1,
        )
    return buffer


def root(tmp_path: Path) -> tuple[Path, Path]:
    scenario = tmp_path / "assets/scenarios/crystal_run_extraction"
    scenario.mkdir(parents=True)
    (scenario / "manifest.json").write_text(
        json.dumps({"wad_sha256": "scenario"}),
        encoding="utf-8",
    )
    cases = tmp_path / "configs/train.json"
    cases.parent.mkdir()
    cases.write_text(
        json.dumps(
            {
                "split": "train",
                "cases": [
                    {
                        "split": "train",
                        "seed": 1,
                        "learner_side": "host",
                        "opponent_style": "strong",
                    },
                    {
                        "split": "train",
                        "seed": 1,
                        "learner_side": "opponent",
                        "opponent_style": "strong",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path, cases


def test_extraction_generation_resumes_from_verified_episode_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, cases = root(tmp_path)
    output = tmp_path / "generated"
    calls = 0

    def interrupted(**kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return episode(), {"host:loot_pickup": 1}

    monkeypatch.setattr(demonstrations, "_collect_episode", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        generate_extraction_demonstrations(
            root=project,
            split="train",
            cases_path=cases,
            output_dir=output,
            style="strong",
            transitions=4,
            shard_size=2,
        )
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    assert progress["transitions"] == 2
    assert len(progress["shards"]) == 1
    assert progress["max_decisions"] == 700
    assert progress["teacher_implementation_sha256"] == (
        demonstrations.extraction_teacher_sha256()
    )

    monkeypatch.setattr(
        demonstrations,
        "_collect_episode",
        lambda **kwargs: (episode(), {"host:loot_pickup": 1}),
    )
    result = generate_extraction_demonstrations(
        root=project,
        split="train",
        cases_path=cases,
        output_dir=output,
        style="strong",
        transitions=4,
        shard_size=2,
        resume=True,
    )

    assert result["transitions"] == 4
    assert len(result["shards"]) == 2
    assert result["test_cases_accessed"] is False
    assert result["teacher_implementation_sha256"] == (
        demonstrations.extraction_teacher_sha256()
    )


def test_extraction_generation_rejects_teacher_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, cases = root(tmp_path)
    output = tmp_path / "generated"
    calls = 0

    def interrupted(**kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return episode(), {}

    monkeypatch.setattr(demonstrations, "_collect_episode", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        generate_extraction_demonstrations(
            root=project,
            split="train",
            cases_path=cases,
            output_dir=output,
            style="strong",
            transitions=4,
            shard_size=2,
        )

    monkeypatch.setattr(
        demonstrations,
        "extraction_teacher_sha256",
        lambda: "0" * 64,
    )
    with pytest.raises(ValueError, match="identity does not match"):
        generate_extraction_demonstrations(
            root=project,
            split="train",
            cases_path=cases,
            output_dir=output,
            style="strong",
            transitions=4,
            shard_size=2,
            resume=True,
        )
