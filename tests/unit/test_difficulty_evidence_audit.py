import hashlib
import json
from pathlib import Path

from botcolosseo.evaluation.difficulty_evidence_audit import (
    audit_difficulty_evidence,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _evidence(root: Path) -> None:
    config = root / "configs/difficulty.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        """
schema_version: 1
profiles:
  easy: {reaction_delay: 2, policy_update_interval: 2}
  normal: {reaction_delay: 1, policy_update_interval: 1}
  hard: {reaction_delay: 0, policy_update_interval: 1}
""".lstrip(),
        encoding="utf-8",
    )
    hashes = {"strong_base": "a" * 64, "aggressive": "b" * 64}
    aggressive = root / "reports/m4/evaluation/aggressive-alpha-025"
    aggressive_summary = aggressive / "summary.json"
    _write(
        aggressive_summary,
        {
            "passed": True,
            "checkpoint_sha256": hashes,
            "test_cases_accessed": False,
        },
    )
    _write(
        aggressive / "manifest.json",
        {
            "passed": True,
            "summary_sha256": _sha(aggressive_summary),
            "scenario_hash": "scenario",
            "test_cases_accessed": False,
        },
    )
    formal = root / "reports/m5/difficulty/formal"
    episodes = formal / "episodes.jsonl"
    episodes.parent.mkdir(parents=True)
    rows = [
        {
            "policy": policy,
            "difficulty": difficulty,
            "opponent": f"opponent-{opponent}",
            "pair_index": pair,
            "learner_side": side,
        }
        for policy in ("strong_base", "aggressive")
        for difficulty in ("easy", "normal", "hard")
        for opponent in range(5)
        for pair in range(10)
        for side in ("host", "opponent")
    ]
    episodes.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = formal / "summary.json"
    _write(
        summary,
        {
            "passed": True,
            "complete": True,
            "episodes": 600,
            "expected_episodes": 600,
            "gates": {"complete": True, "monotonic": True},
            "stage": "m5-difficulty",
            "checkpoint_sha256": hashes,
            "config_sha256": _sha(config),
            "test_cases_accessed": False,
        },
    )
    _write(
        formal / "manifest.json",
        {
            "passed": True,
            "episodes": 600,
            "episodes_sha256": _sha(episodes),
            "summary_sha256": _sha(summary),
            "scenario_hash": "scenario",
            "config_sha256": _sha(config),
            "profiles": {
                "easy": {
                    "reaction_delay": 2,
                    "policy_update_interval": 2,
                },
                "normal": {
                    "reaction_delay": 1,
                    "policy_update_interval": 1,
                },
                "hard": {
                    "reaction_delay": 0,
                    "policy_update_interval": 1,
                },
            },
            "test_cases_accessed": False,
        },
    )


def test_difficulty_audit_accepts_complete_hash_chain(tmp_path: Path) -> None:
    _evidence(tmp_path)

    result = audit_difficulty_evidence(tmp_path)

    assert result["passed"] is True
    assert all(result["checks"].values())


def test_difficulty_audit_rejects_tampered_ledger(tmp_path: Path) -> None:
    _evidence(tmp_path)
    episodes = tmp_path / "reports/m5/difficulty/formal/episodes.jsonl"
    episodes.write_text(episodes.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    result = audit_difficulty_evidence(tmp_path)

    assert result["passed"] is False
    assert result["checks"]["formal_artifact_hashes"] is False
