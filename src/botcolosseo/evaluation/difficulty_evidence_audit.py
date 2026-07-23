from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from botcolosseo.agents.difficulty import load_difficulty_profiles
from botcolosseo.agents.league_opponents import sha256_file


@dataclass
class _Audit:
    checks: dict[str, bool]
    errors: list[str]

    def add(self, name: str, condition: bool, message: str) -> None:
        self.checks[name] = bool(condition)
        if not condition:
            self.errors.append(message)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def audit_difficulty_evidence(root: Path) -> dict[str, object]:
    root = root.expanduser().resolve()
    paths = {
        "config": root / "configs/difficulty.yaml",
        "aggressive_manifest": root
        / "reports/m4/evaluation/aggressive-alpha-025/manifest.json",
        "aggressive_summary": root
        / "reports/m4/evaluation/aggressive-alpha-025/summary.json",
        "formal_manifest": root / "reports/m5/difficulty/formal/manifest.json",
        "formal_summary": root / "reports/m5/difficulty/formal/summary.json",
        "formal_episodes": root / "reports/m5/difficulty/formal/episodes.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing M5 difficulty evidence: " + ", ".join(missing))
    profiles = load_difficulty_profiles(paths["config"])
    aggressive_manifest = _json(paths["aggressive_manifest"])
    aggressive_summary = _json(paths["aggressive_summary"])
    formal_manifest = _json(paths["formal_manifest"])
    formal_summary = _json(paths["formal_summary"])
    audit = _Audit({}, [])

    expected_profiles = {
        name: {
            "reaction_delay": profile.reaction_delay,
            "policy_update_interval": profile.policy_update_interval,
        }
        for name, profile in profiles.items()
    }
    audit.add(
        "config_identity",
        formal_manifest.get("config_sha256") == sha256_file(paths["config"])
        and formal_summary.get("config_sha256") == sha256_file(paths["config"])
        and formal_manifest.get("profiles") == expected_profiles,
        "Frozen difficulty config identity is invalid",
    )
    aggressive_hashes = aggressive_summary.get("checkpoint_sha256")
    audit.add(
        "aggressive_upstream_gate",
        aggressive_manifest.get("passed") is True
        and aggressive_summary.get("passed") is True
        and isinstance(aggressive_hashes, dict)
        and aggressive_manifest.get("summary_sha256")
        == sha256_file(paths["aggressive_summary"])
        and aggressive_manifest.get("test_cases_accessed") is False
        and aggressive_summary.get("test_cases_accessed") is False,
        "Aggressive upstream gate or hash chain is invalid",
    )

    episodes_path = paths["formal_episodes"]
    summary_path = paths["formal_summary"]
    rows = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identities = {
        (
            row.get("policy"),
            row.get("difficulty"),
            row.get("opponent"),
            row.get("pair_index"),
            row.get("learner_side"),
        )
        for row in rows
        if isinstance(row, dict)
    }
    audit.add(
        "formal_artifact_hashes",
        formal_manifest.get("episodes") == 600
        and len(rows) == 600
        and len(identities) == 600
        and formal_manifest.get("episodes_sha256") == sha256_file(episodes_path)
        and formal_manifest.get("summary_sha256") == sha256_file(summary_path),
        "Formal difficulty ledger is incomplete, duplicated, or hash-mismatched",
    )
    formal_hashes = formal_summary.get("checkpoint_sha256")
    formal_gates = formal_summary.get("gates")
    audit.add(
        "formal_gate",
        formal_manifest.get("passed") is True
        and formal_summary.get("passed") is True
        and formal_summary.get("complete") is True
        and formal_summary.get("episodes") == 600
        and formal_summary.get("expected_episodes") == 600
        and isinstance(formal_gates, dict)
        and bool(formal_gates)
        and all(value is True for value in formal_gates.values())
        and isinstance(formal_hashes, dict)
        and isinstance(aggressive_hashes, dict)
        and formal_hashes == aggressive_hashes,
        "Formal difficulty gate or checkpoint identity is invalid",
    )
    scenario = aggressive_manifest.get("scenario_hash")
    audit.add(
        "scenario_identity",
        bool(scenario)
        and formal_manifest.get("scenario_hash") == scenario
        and formal_summary.get("stage") == "m5-difficulty",
        "M5 difficulty scenario identity is inconsistent",
    )
    audit.add(
        "no_test_access",
        all(
            payload.get("test_cases_accessed") is False
            for payload in (
                aggressive_manifest,
                aggressive_summary,
                formal_manifest,
                formal_summary,
            )
        ),
        "M5 difficulty evidence does not prove zero test-case access",
    )
    return {
        "schema_version": 1,
        "stage": "m5-difficulty-audit",
        "passed": all(audit.checks.values()),
        "checks": audit.checks,
        "errors": audit.errors,
        "config_sha256": sha256_file(paths["config"]),
        "checkpoint_sha256": formal_hashes,
        "test_cases_accessed": False,
    }
