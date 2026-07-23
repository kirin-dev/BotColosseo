from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.evaluation.defensive import PROTECTIVE_PRESENCE_ESTIMATOR


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


def audit_defensive_ppo_evidence(root: Path) -> dict[str, object]:
    root = root.expanduser().resolve()
    paths = {
        "warm_report": root / "runs/m5/defensive-interpolation/alpha-025.json",
        "warm_checkpoint": root / "runs/m5/defensive-interpolation/alpha-025.pt",
        "training": root / "runs/m5/defensive-ppo-main/summary.json",
        "candidate": root
        / "runs/m5/defensive-ppo-main/candidate-0200000.pt",
        "smoke_manifest": root
        / "reports/m5/defensive/ppo-repair/smoke/manifest.json",
        "smoke_summary": root
        / "reports/m5/defensive/ppo-repair/smoke/summary.json",
        "formal_manifest": root
        / "reports/m5/defensive/ppo-repair/formal/manifest.json",
        "formal_summary": root
        / "reports/m5/defensive/ppo-repair/formal/summary.json",
        "formal_episodes": root
        / "reports/m5/defensive/ppo-repair/formal/episodes.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Defensive PPO evidence: " + ", ".join(missing)
        )
    warm = _json(paths["warm_report"])
    training = _json(paths["training"])
    smoke_manifest = _json(paths["smoke_manifest"])
    smoke_summary = _json(paths["smoke_summary"])
    formal_manifest = _json(paths["formal_manifest"])
    formal_summary = _json(paths["formal_summary"])
    audit = _Audit({}, [])

    warm_hash = sha256_file(paths["warm_checkpoint"])
    audit.add(
        "warm_start_identity",
        warm.get("style") == "defensive"
        and warm.get("alpha") == 0.25
        and warm.get("checkpoint_sha256") == warm_hash
        and warm.get("test_cases_accessed") is False,
        "Defensive PPO warm-start identity is invalid",
    )
    candidate_hash = sha256_file(paths["candidate"])
    candidates = training.get("candidate_checkpoints")
    audit.add(
        "training_gate",
        training.get("completed") is True
        and training.get("environment_steps") == 200_000
        and training.get("style") == "defensive"
        and training.get("style_warm_start_sha256") == warm_hash
        and training.get("test_cases_accessed") is False
        and isinstance(candidates, list)
        and any(
            isinstance(entry, dict)
            and entry.get("environment_steps") == 200_000
            and entry.get("sha256") == candidate_hash
            for entry in candidates
        ),
        "Defensive PPO training or candidate identity is invalid",
    )
    audit.add(
        "smoke_gate",
        smoke_manifest.get("passed") is True
        and smoke_manifest.get("episodes") == 20
        and smoke_summary.get("passed") is True
        and smoke_summary.get("protective_presence_estimator")
        == PROTECTIVE_PRESENCE_ESTIMATOR
        and smoke_summary.get("checkpoint_sha256", {}).get("defensive")
        == candidate_hash
        and smoke_manifest.get("summary_sha256")
        == sha256_file(paths["smoke_summary"]),
        "Defensive PPO smoke gate or checkpoint identity is invalid",
    )

    rows = [
        json.loads(line)
        for line in paths["formal_episodes"].read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    identities = {
        (
            row.get("policy"),
            row.get("opponent"),
            row.get("pair_index"),
            row.get("learner_side"),
        )
        for row in rows
        if isinstance(row, dict)
    }
    audit.add(
        "formal_artifact_hashes",
        formal_manifest.get("episodes") == 200
        and len(rows) == 200
        and len(identities) == 200
        and formal_manifest.get("episodes_sha256")
        == sha256_file(paths["formal_episodes"])
        and formal_manifest.get("summary_sha256")
        == sha256_file(paths["formal_summary"]),
        "Formal Defensive PPO ledger is incomplete or hash-mismatched",
    )
    gates = formal_summary.get("gates")
    formal_hashes = formal_summary.get("checkpoint_sha256")
    audit.add(
        "formal_gate",
        formal_manifest.get("passed") is True
        and formal_summary.get("passed") is True
        and formal_summary.get("complete") is True
        and formal_summary.get("protective_presence_estimator")
        == PROTECTIVE_PRESENCE_ESTIMATOR
        and isinstance(gates, dict)
        and bool(gates)
        and all(value is True for value in gates.values())
        and isinstance(formal_hashes, dict)
        and formal_hashes.get("defensive") == candidate_hash
        and formal_hashes.get("strong_base")
        == training.get("base_checkpoint_sha256"),
        "Formal Defensive PPO gate or selected checkpoint identity is invalid",
    )
    scenario = training.get("scenario_hash")
    audit.add(
        "scenario_identity",
        bool(scenario)
        and warm.get("scenario_hash") == scenario
        and smoke_manifest.get("scenario_hash") == scenario
        and formal_manifest.get("scenario_hash") == scenario,
        "Defensive PPO scenario identity is inconsistent",
    )
    audit.add(
        "no_test_access",
        all(
            payload.get("test_cases_accessed") is False
            for payload in (
                warm,
                training,
                smoke_manifest,
                smoke_summary,
                formal_manifest,
                formal_summary,
            )
        ),
        "Defensive PPO evidence does not prove zero test-case access",
    )
    return {
        "schema_version": 1,
        "stage": "m5-defensive-ppo-audit",
        "passed": all(audit.checks.values()),
        "checks": audit.checks,
        "errors": audit.errors,
        "warm_start_sha256": warm_hash,
        "candidate_checkpoint_sha256": candidate_hash,
        "test_cases_accessed": False,
    }
