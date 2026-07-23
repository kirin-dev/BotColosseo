from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _resolve(root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def audit_defensive_evidence(root: Path) -> dict[str, object]:
    root = root.expanduser().resolve()
    paths = {
        "data": root / "data/generated/m5/defensive/train-manifest.json",
        "distillation": root / "runs/m5/defensive-distillation/summary.json",
        "selection": root / "reports/m5/defensive/selection.json",
        "formal_manifest": root / "reports/m5/defensive/formal/manifest.json",
        "formal_summary": root / "reports/m5/defensive/formal/summary.json",
        "formal_episodes": root / "reports/m5/defensive/formal/episodes.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing M5 Defensive evidence: " + ", ".join(missing))
    data = _json(paths["data"])
    distillation = _json(paths["distillation"])
    selection = _json(paths["selection"])
    formal_manifest = _json(paths["formal_manifest"])
    formal_summary = _json(paths["formal_summary"])
    audit = _Audit({}, [])

    data_gates = data.get("gate")
    audit.add(
        "data_gate",
        data.get("passed") is True
        and data.get("style") == "defensive"
        and data.get("split") == "train"
        and data.get("transitions") == 50_000
        and isinstance(data_gates, dict)
        and bool(data_gates)
        and all(value is True for value in data_gates.values()),
        "Production Defensive data gate is incomplete or failed",
    )
    shards = data.get("shards")
    shard_hashes_valid = isinstance(shards, list) and len(shards) == 5
    if shard_hashes_valid:
        for entry in shards:
            if not isinstance(entry, dict):
                shard_hashes_valid = False
                break
            shard = paths["data"].parent / str(entry.get("file", ""))
            if (
                not shard.is_file()
                or entry.get("sha256") != sha256_file(shard)
                or entry.get("transitions") != 10_000
            ):
                shard_hashes_valid = False
                break
    audit.add(
        "data_shard_hashes", shard_hashes_valid, "Defensive data shard hash chain is invalid"
    )

    base_hash = data.get("base_checkpoint_sha256")
    distill_checkpoint = _resolve(root, distillation.get("checkpoint"))
    neutral_checkpoint = _resolve(root, distillation.get("neutral_checkpoint"))
    audit.add(
        "distillation_gate",
        distillation.get("passed") is True
        and distillation.get("completed") is True
        and distillation.get("style") == "defensive"
        and distillation.get("frozen_base") is True
        and distillation.get("base_checkpoint_sha256") == base_hash
        and distillation.get("data_manifest_sha256") == sha256_file(paths["data"])
        and distillation.get("offline_evaluation", {}).get("passed") is True,
        "Defensive distillation gate or its upstream identity is invalid",
    )
    audit.add(
        "distillation_checkpoint_hashes",
        distill_checkpoint is not None
        and distill_checkpoint.is_file()
        and distillation.get("checkpoint_sha256") == sha256_file(distill_checkpoint)
        and neutral_checkpoint is not None
        and neutral_checkpoint.is_file()
        and distillation.get("neutral_checkpoint_sha256")
        == sha256_file(neutral_checkpoint),
        "Defensive distillation checkpoint hashes are invalid",
    )

    selected = selection.get("selected")
    selected_checkpoint = (
        _resolve(root, selected.get("checkpoint")) if isinstance(selected, dict) else None
    )
    audit.add(
        "selection_gate",
        selection.get("passed") is True
        and selection.get("grid") == [0.25, 0.5, 0.75]
        and isinstance(selected, dict)
        and selected.get("eligible") is True
        and selected_checkpoint is not None
        and selected_checkpoint.is_file()
        and selected.get("checkpoint_sha256") == sha256_file(selected_checkpoint),
        "Frozen Defensive alpha selection is invalid",
    )

    summary_path = paths["formal_summary"]
    episodes_path = paths["formal_episodes"]
    episode_lines = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    episode_keys = {
        (
            row.get("policy"),
            row.get("opponent"),
            row.get("pair_index"),
            row.get("learner_side"),
        )
        for row in episode_lines
        if isinstance(row, dict)
    }
    audit.add(
        "formal_artifact_hashes",
        formal_manifest.get("episodes_sha256") == sha256_file(episodes_path)
        and formal_manifest.get("summary_sha256") == sha256_file(summary_path)
        and formal_manifest.get("episodes") == 200
        and len(episode_lines) == 200
        and len(episode_keys) == 200,
        "Formal Defensive ledger is incomplete, duplicated, or hash-mismatched",
    )
    formal_gates = formal_summary.get("gates")
    selected_hash = selected.get("checkpoint_sha256") if isinstance(selected, dict) else None
    checkpoint_hashes = formal_summary.get("checkpoint_sha256")
    audit.add(
        "formal_gate",
        formal_manifest.get("passed") is True
        and formal_summary.get("passed") is True
        and formal_summary.get("complete") is True
        and formal_summary.get("episodes") == 200
        and formal_summary.get("expected_episodes") == 200
        and isinstance(formal_gates, dict)
        and bool(formal_gates)
        and all(value is True for value in formal_gates.values())
        and isinstance(checkpoint_hashes, dict)
        and checkpoint_hashes.get("defensive") == selected_hash
        and checkpoint_hashes.get("strong_base") == base_hash,
        "Formal Defensive gate or selected checkpoint identity is invalid",
    )
    scenario = data.get("scenario_hash")
    audit.add(
        "scenario_identity",
        bool(scenario)
        and distillation.get("scenario_hash") == scenario
        and formal_manifest.get("scenario_hash") == scenario,
        "M5 Defensive scenario identity is inconsistent",
    )
    audit.add(
        "no_test_access",
        all(
            payload.get("test_cases_accessed") is False
            for payload in (data, distillation, selection, formal_manifest, formal_summary)
        ),
        "M5 Defensive evidence does not prove zero test-case access",
    )
    return {
        "schema_version": 1,
        "stage": "m5-defensive-audit",
        "passed": all(audit.checks.values()),
        "checks": audit.checks,
        "errors": audit.errors,
        "selected_alpha": selected.get("alpha") if isinstance(selected, dict) else None,
        "selected_checkpoint_sha256": selected_hash,
        "test_cases_accessed": False,
    }
