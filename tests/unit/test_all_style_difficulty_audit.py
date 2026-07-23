from __future__ import annotations

import hashlib
import json
from pathlib import Path

from botcolosseo.evaluation.all_style_difficulty_audit import (
    audit_all_style_difficulty,
)
from botcolosseo.evaluation.defensive import PROTECTIVE_PRESENCE_ESTIMATOR
from botcolosseo.evaluation.explorer import ROUTE_ENTROPY_ESTIMATOR

_BLOCKS = {
    "aggressive": (
        "reports/m5/difficulty/formal",
        "reports/m4/evaluation/aggressive-alpha-025",
        "m5-difficulty",
        "m4",
        None,
    ),
    "defensive": (
        "reports/m5/defensive/difficulty/formal",
        "reports/m5/defensive/ppo-repair/formal",
        "m5-defensive-difficulty",
        "m5-defensive",
        PROTECTIVE_PRESENCE_ESTIMATOR,
    ),
    "explorer": (
        "reports/m5/explorer/difficulty/formal",
        "reports/m5/explorer/ppo-repair/formal",
        "m5-explorer-difficulty",
        "m5-explorer",
        ROUTE_ENTROPY_ESTIMATOR,
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _rows(style: str) -> list[dict[str, object]]:
    return [
        {
            "policy": policy,
            "difficulty": difficulty,
            "opponent": f"opponent-{opponent}",
            "pair_index": pair,
            "learner_side": side,
            "seed": 1000 + opponent * 20 + pair * 2 + (side == "opponent"),
            "outcome": "win",
            "objective_completed": True,
            "learner_score": 3,
            "opponent_score": 0,
            "decisions": 20,
            "terminated": True,
            "truncated": False,
            "peer_tic_lag_max": 0,
            "protocol_inconsistent": False,
            "action_tic_inconsistent": False,
            "score_event_inconsistent": False,
            "scenario_hash": "scenario",
        }
        for policy in ("strong_base", style)
        for difficulty in ("easy", "normal", "hard")
        for opponent in range(5)
        for pair in range(10)
        for side in ("host", "opponent")
    ]


def _evidence(root: Path) -> None:
    base = "1" * 64
    style_hashes = {
        "aggressive": "2" * 64,
        "defensive": "3" * 64,
        "explorer": "4" * 64,
    }
    profiles = {
        "easy": {"reaction_delay": 2, "policy_update_interval": 2},
        "normal": {"reaction_delay": 1, "policy_update_interval": 1},
        "hard": {"reaction_delay": 0, "policy_update_interval": 1},
    }
    config = root / "configs/difficulty.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("frozen: true\n", encoding="utf-8")
    cases = root / "configs/m2/validation.json"
    cases.parent.mkdir(parents=True)
    cases.write_text("[]\n", encoding="utf-8")
    scenario_manifest = root / "assets/scenarios/crystal_run/manifest.json"
    _write(scenario_manifest, {"wad_sha256": "scenario"})
    selected = [
        [f"opponent-{opponent}", pair, side]
        for opponent in range(5)
        for pair in range(10)
        for side in ("host", "opponent")
    ]
    for style, (
        directory_name,
        upstream_name,
        stage,
        upstream_stage,
        estimator,
    ) in _BLOCKS.items():
        hashes = {"strong_base": base, style: style_hashes[style]}
        upstream = root / upstream_name
        upstream_summary = upstream / "summary.json"
        _write(
            upstream_summary,
            {
                "stage": upstream_stage,
                "split": "validation",
                "passed": True,
                "complete": True,
                "checkpoint_sha256": hashes,
                "test_cases_accessed": False,
            },
        )
        _write(
            upstream / "manifest.json",
            {
                "passed": True,
                "summary_sha256": _sha(upstream_summary),
                "test_cases_accessed": False,
            },
        )

        directory = root / directory_name
        episodes = directory / "episodes.jsonl"
        episodes.parent.mkdir(parents=True, exist_ok=True)
        episodes.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in _rows(style)
            ),
            encoding="utf-8",
        )
        summary = directory / "summary.json"
        summary_payload = {
            "stage": stage,
            "split": "validation",
            "passed": True,
            "complete": True,
            "episodes": 600,
            "expected_episodes": 600,
            "protocol_inconsistencies": 0,
            "gates": {"complete": True, "style": True},
            "cells": {
                "strong_base": {
                    difficulty: {
                        "episodes": 100,
                        "performance": performance,
                        "objective_rate": performance,
                    }
                    for difficulty, performance in (
                        ("easy", 0.8),
                        ("normal", 0.9),
                        ("hard", 1.0),
                    )
                },
                style: {
                    difficulty: {
                        "episodes": 100,
                        "performance": performance,
                        "objective_rate": performance,
                    }
                    for difficulty, performance in (
                        ("easy", 0.8),
                        ("normal", 0.9),
                        ("hard", 1.0),
                    )
                },
            },
            "checkpoint_sha256": hashes,
            "config_sha256": _sha(config),
            "scenario_hash": "scenario",
            "test_cases_accessed": False,
        }
        if estimator is not None:
            summary_payload["estimator"] = estimator
        _write(summary, summary_payload)
        manifest_payload = {
            "passed": True,
            "episodes": 600,
            "expected_episodes": 600,
            "episodes_sha256": _sha(episodes),
            "summary_sha256": _sha(summary),
            "checkpoint_sha256": hashes,
            "config_sha256": _sha(config),
            "cases_sha256": _sha(cases),
            "scenario_hash": "scenario",
            "selected_case_ids": selected,
            "profiles": profiles,
            "pairs_per_opponent": 10,
            "max_decisions": 525,
            "max_attempts": 2,
            "test_cases_accessed": False,
        }
        if estimator is not None:
            manifest_payload["estimator"] = estimator
        _write(directory / "manifest.json", manifest_payload)


def test_all_style_audit_accepts_1800_hash_bound_episodes(tmp_path: Path) -> None:
    _evidence(tmp_path)

    result = audit_all_style_difficulty(tmp_path)

    assert result["passed"] is True
    assert result["episodes"] == 1800
    assert result["checkpoint_sha256"] == {
        "strong_base": "1" * 64,
        "aggressive": "2" * 64,
        "defensive": "3" * 64,
        "explorer": "4" * 64,
    }
    assert tuple(result["cells"]) == (
        "strong_base",
        "aggressive",
        "defensive",
        "explorer",
    )
    assert all(result["gates"].values())


def test_all_style_audit_rejects_strong_base_outcome_drift(tmp_path: Path) -> None:
    _evidence(tmp_path)
    directory = tmp_path / "reports/m5/explorer/difficulty/formal"
    episodes = directory / "episodes.jsonl"
    rows = [
        json.loads(line)
        for line in episodes.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["learner_score"] = 2
    episodes.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["episodes_sha256"] = _sha(episodes)
    _write(directory / "manifest.json", manifest)

    result = audit_all_style_difficulty(tmp_path)

    assert result["passed"] is False
    assert result["gates"]["strong_base_determinism"] is False


def test_all_style_audit_rejects_tampered_block_hash(tmp_path: Path) -> None:
    _evidence(tmp_path)
    episodes = tmp_path / "reports/m5/defensive/difficulty/formal/episodes.jsonl"
    episodes.write_text(
        episodes.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )

    result = audit_all_style_difficulty(tmp_path)

    assert result["passed"] is False
    assert result["gates"]["defensive_artifact_chain"] is False
