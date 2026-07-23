from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.evaluation.defensive import PROTECTIVE_PRESENCE_ESTIMATOR
from botcolosseo.evaluation.explorer import ROUTE_ENTROPY_ESTIMATOR

_BLOCKS = {
    "aggressive": {
        "stage": "m5-difficulty",
        "directory": "reports/m5/difficulty/formal",
        "upstream": "reports/m4/evaluation/aggressive-alpha-025",
        "upstream_stage": "m4",
        "estimator": None,
    },
    "defensive": {
        "stage": "m5-defensive-difficulty",
        "directory": "reports/m5/defensive/difficulty/formal",
        "upstream": "reports/m5/defensive/ppo-repair/formal",
        "upstream_stage": "m5-defensive",
        "estimator": PROTECTIVE_PRESENCE_ESTIMATOR,
    },
    "explorer": {
        "stage": "m5-explorer-difficulty",
        "directory": "reports/m5/explorer/difficulty/formal",
        "upstream": "reports/m5/explorer/ppo-repair/formal",
        "upstream_stage": "m5-explorer",
        "estimator": ROUTE_ENTROPY_ESTIMATOR,
    },
}
_COMMON_OUTCOME_FIELDS = (
    "seed",
    "outcome",
    "objective_completed",
    "learner_score",
    "opponent_score",
    "decisions",
    "terminated",
    "truncated",
    "peer_tic_lag_max",
    "protocol_inconsistent",
    "action_tic_inconsistent",
    "score_event_inconsistent",
    "scenario_hash",
)


@dataclass
class _Audit:
    checks: dict[str, bool]
    errors: list[str]

    def add(self, name: str, condition: bool, message: str) -> None:
        self.checks[name] = bool(condition)
        if not condition:
            self.errors.append(message)


def audit_all_style_difficulty(root: Path) -> dict[str, object]:
    root = root.expanduser().resolve()
    evidence = _load_evidence(root)
    audit = _Audit({}, [])

    manifests = {
        style: payload["manifest"] for style, payload in evidence.items()
    }
    summaries = {
        style: payload["summary"] for style, payload in evidence.items()
    }
    rows = {style: payload["rows"] for style, payload in evidence.items()}
    upstream_manifests = {
        style: payload["upstream_manifest"] for style, payload in evidence.items()
    }
    upstream_summaries = {
        style: payload["upstream_summary"] for style, payload in evidence.items()
    }

    for style, settings in _BLOCKS.items():
        manifest = manifests[style]
        summary = summaries[style]
        block_rows = rows[style]
        identities = {_identity(row) for row in block_rows}
        gates = summary.get("gates")
        audit.add(
            f"{style}_artifact_chain",
            len(block_rows) == 600
            and len(identities) == 600
            and manifest.get("episodes") == 600
            and manifest.get("expected_episodes") == 600
            and manifest.get("episodes_sha256")
            == sha256_file(evidence[style]["episodes_path"])
            and manifest.get("summary_sha256")
            == sha256_file(evidence[style]["summary_path"]),
            f"{style} difficulty ledger is incomplete, duplicated, or hash-mismatched",
        )
        audit.add(
            f"{style}_formal_gate",
            manifest.get("passed") is True
            and summary.get("stage") == settings["stage"]
            and summary.get("split") == "validation"
            and summary.get("passed") is True
            and summary.get("complete") is True
            and summary.get("episodes") == 600
            and summary.get("expected_episodes") == 600
            and summary.get("protocol_inconsistencies") == 0
            and isinstance(gates, dict)
            and bool(gates)
            and all(value is True for value in gates.values()),
            f"{style} difficulty formal gate is invalid",
        )
        upstream_manifest = upstream_manifests[style]
        upstream_summary = upstream_summaries[style]
        audit.add(
            f"{style}_upstream_gate",
            upstream_manifest.get("passed") is True
            and upstream_manifest.get("summary_sha256")
            == sha256_file(evidence[style]["upstream_summary_path"])
            and upstream_summary.get("stage") == settings["upstream_stage"]
            and upstream_summary.get("split") == "validation"
            and upstream_summary.get("passed") is True
            and upstream_summary.get("complete") is True
            and upstream_summary.get("test_cases_accessed") is False
            and upstream_summary.get("checkpoint_sha256")
            == summary.get("checkpoint_sha256"),
            f"{style} upstream style gate or checkpoint binding is invalid",
        )
        estimator = settings["estimator"]
        audit.add(
            f"{style}_estimator_identity",
            estimator is None
            or (
                manifest.get("estimator") == estimator
                and summary.get("estimator") == estimator
            ),
            f"{style} difficulty estimator identity is invalid",
        )

    reference = manifests["aggressive"]
    audit.add(
        "shared_protocol_identity",
        all(
            manifest.get("config_sha256") == reference.get("config_sha256")
            and manifest.get("cases_sha256") == reference.get("cases_sha256")
            and manifest.get("scenario_hash") == reference.get("scenario_hash")
            and manifest.get("selected_case_ids")
            == reference.get("selected_case_ids")
            and manifest.get("profiles") == reference.get("profiles")
            and manifest.get("pairs_per_opponent") == 10
            and manifest.get("max_decisions") == 525
            and manifest.get("max_attempts") == 2
            for manifest in manifests.values()
        ),
        "Difficulty blocks do not share the frozen protocol identity",
    )
    audit.add(
        "summary_manifest_identity",
        all(
            summaries[style].get("checkpoint_sha256")
            == manifests[style].get("checkpoint_sha256")
            and summaries[style].get("config_sha256")
            == manifests[style].get("config_sha256")
            and summaries[style].get("scenario_hash")
            == manifests[style].get("scenario_hash")
            for style in _BLOCKS
        ),
        "Difficulty summary and manifest identities are inconsistent",
    )
    checkpoints = _combined_checkpoints(summaries)
    audit.add(
        "checkpoint_identity",
        checkpoints is not None,
        "Difficulty blocks do not share one Strong Base or contain all style checkpoints",
    )
    combined_identities = {
        (style, *_identity(row))
        for style, block_rows in rows.items()
        for row in block_rows
    }
    audit.add(
        "combined_episode_identity",
        sum(len(block_rows) for block_rows in rows.values()) == 1800
        and len(combined_identities) == 1800,
        "All-style difficulty evidence is not exactly 1,800 unique episodes",
    )
    audit.add(
        "strong_base_determinism",
        _strong_base_outcomes_match(rows),
        "Strong Base common outcomes differ across style-specific ledgers",
    )
    audit.add(
        "no_test_access",
        all(
            payload.get("test_cases_accessed") is False
            for style in _BLOCKS
            for payload in (
                manifests[style],
                summaries[style],
                upstream_manifests[style],
                upstream_summaries[style],
            )
        ),
        "All-style difficulty evidence does not prove zero test-case access",
    )
    passed = all(audit.checks.values())
    return {
        "schema_version": 1,
        "stage": "m5-all-style-difficulty",
        "split": "validation",
        "complete": passed,
        "passed": passed,
        "episodes": 1800,
        "expected_episodes": 1800,
        "protocol_inconsistencies": 0 if passed else None,
        "gates": audit.checks,
        "errors": audit.errors,
        "checkpoint_sha256": checkpoints or {},
        "block_summary_sha256": {
            style: sha256_file(evidence[style]["summary_path"])
            for style in _BLOCKS
        },
        "config_sha256": reference.get("config_sha256"),
        "cases_sha256": reference.get("cases_sha256"),
        "scenario_hash": reference.get("scenario_hash"),
        "test_cases_accessed": False,
    }


def _load_evidence(root: Path) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for style, settings in _BLOCKS.items():
        directory = root / str(settings["directory"])
        upstream = root / str(settings["upstream"])
        paths = {
            "manifest_path": directory / "manifest.json",
            "summary_path": directory / "summary.json",
            "episodes_path": directory / "episodes.jsonl",
            "upstream_manifest_path": upstream / "manifest.json",
            "upstream_summary_path": upstream / "summary.json",
        }
        missing.extend(str(path) for path in paths.values() if not path.is_file())
        evidence[style] = paths
    if missing:
        raise FileNotFoundError(
            "Missing all-style difficulty evidence: " + ", ".join(missing)
        )
    for payload in evidence.values():
        payload["manifest"] = _json(payload["manifest_path"])
        payload["summary"] = _json(payload["summary_path"])
        payload["upstream_manifest"] = _json(payload["upstream_manifest_path"])
        payload["upstream_summary"] = _json(payload["upstream_summary_path"])
        payload["rows"] = _jsonl(payload["episodes_path"])
    return evidence


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects in ledger: {path}")
    return rows


def _identity(row: dict[str, Any]) -> tuple[object, ...]:
    return (
        row.get("policy"),
        row.get("difficulty"),
        row.get("opponent"),
        row.get("pair_index"),
        row.get("learner_side"),
    )


def _combined_checkpoints(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, str] | None:
    hashes = {
        style: summary.get("checkpoint_sha256")
        for style, summary in summaries.items()
    }
    if any(not isinstance(value, dict) for value in hashes.values()):
        return None
    base = hashes["aggressive"].get("strong_base")
    if not isinstance(base, str) or any(
        value.get("strong_base") != base for value in hashes.values()
    ):
        return None
    result = {"strong_base": base}
    for style in _BLOCKS:
        digest = hashes[style].get(style)
        if not isinstance(digest, str):
            return None
        result[style] = digest
    return result


def _strong_base_outcomes_match(
    rows: dict[str, list[dict[str, Any]]],
) -> bool:
    projections: dict[str, dict[tuple[object, ...], tuple[object, ...]]] = {}
    for style, block_rows in rows.items():
        selected = [row for row in block_rows if row.get("policy") == "strong_base"]
        projections[style] = {
            _identity(row)[1:]: tuple(row.get(field) for field in _COMMON_OUTCOME_FIELDS)
            for row in selected
        }
        if len(selected) != 300 or len(projections[style]) != 300:
            return False
    return (
        projections["aggressive"]
        == projections["defensive"]
        == projections["explorer"]
    )
