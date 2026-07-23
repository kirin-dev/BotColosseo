from __future__ import annotations

import json
from pathlib import Path

import pytest

import botcolosseo.evaluation.checkpoint_release as release
from botcolosseo.agents.league_opponents import sha256_file


def _sources(root: Path) -> dict[str, Path]:
    result = {}
    for index, policy_id in enumerate(release.M6_POLICIES):
        path = root / f"{policy_id}.pt"
        path.write_bytes(f"checkpoint-{index}".encode())
        result[policy_id] = path
    return result


def _metrics(path: Path, sources: dict[str, Path]) -> None:
    payload = {
        "schema_version": 2,
        "stage": "m6",
        "split": "validation",
        "passed": True,
        "style_gate_passed": True,
        "retention_gate_passed": True,
        "difficulty_gate_passed": True,
        "episodes": 1800,
        "checkpoint_sha256": {
            policy_id: sha256_file(source)
            for policy_id, source in sources.items()
        },
        "headline_cards": [
            {"label": f"Metric {index}", "value": str(index)}
            for index in range(6)
        ],
        "case_contrast_scores": {"case": 1.0},
        "decision_contrast_scores": {"case": [0.0, 1.0]},
        "upstream_sha256": {
            "m4": "1" * 64,
            "defensive": "2" * 64,
            "explorer": "3" * 64,
            "difficulty": "4" * 64,
        },
        "test_cases_accessed": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_checkpoint_release_preserves_verified_source_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _sources(tmp_path)
    metrics = tmp_path / "metrics.json"
    _metrics(metrics, sources)
    verified = []
    monkeypatch.setattr(
        release.CheckpointOpponentPolicy,
        "load",
        lambda spec, device: verified.append((spec.opponent_id, str(device))),
    )

    manifest = release.build_checkpoint_release(
        metrics_path=metrics,
        sources=sources,
        scenario_hash="a" * 64,
        output_dir=tmp_path / "release",
    )

    assert manifest["fair_observation_loader_verified"] is True
    assert manifest["optimizer_state_stripped"] is False
    assert [row[0] for row in verified] == list(release.M6_POLICIES)
    for row in manifest["policies"]:
        target = tmp_path / "release" / row["path"]
        assert sha256_file(target) == row["sha256"]


def test_checkpoint_release_rejects_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _sources(tmp_path)
    metrics = tmp_path / "metrics.json"
    _metrics(metrics, sources)
    sources["explorer"].write_bytes(b"tampered")
    monkeypatch.setattr(
        release.CheckpointOpponentPolicy,
        "load",
        lambda spec, device: object(),
    )

    with pytest.raises(ValueError, match="explorer"):
        release.build_checkpoint_release(
            metrics_path=metrics,
            sources=sources,
            scenario_hash="a" * 64,
            output_dir=tmp_path / "release",
        )
    assert not (tmp_path / "release").exists()


def test_checkpoint_release_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _sources(tmp_path)
    metrics = tmp_path / "metrics.json"
    _metrics(metrics, sources)
    monkeypatch.setattr(
        release.CheckpointOpponentPolicy,
        "load",
        lambda spec, device: object(),
    )
    output = tmp_path / "release"
    release.build_checkpoint_release(
        metrics_path=metrics,
        sources=sources,
        scenario_hash="a" * 64,
        output_dir=output,
    )

    with pytest.raises(FileExistsError, match="overwrite"):
        release.build_checkpoint_release(
            metrics_path=metrics,
            sources=sources,
            scenario_hash="a" * 64,
            output_dir=output,
        )
