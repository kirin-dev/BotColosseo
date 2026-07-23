from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from botcolosseo.agents.difficulty import DIFFICULTIES
from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.evaluation.difficulty import MONOTONIC_TOLERANCE
from botcolosseo.evaluation.hybrid_difficulty_config import (
    HybridDifficultyProductConfig,
    load_hybrid_difficulty_product_config,
)
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS

POLICIES = ("strong_base", "aggressive", "defensive", "explorer")


def audit_hybrid_all_style_difficulty(
    root: Path,
    *,
    config_path: Path = Path("configs/m5/hybrid/difficulty-product.yaml"),
    defensive_extension: Path = Path(
        "reports/m5/hybrid/defensive/difficulty/formal"
    ),
    explorer_extension: Path = Path(
        "reports/m5/hybrid/explorer/difficulty/formal"
    ),
) -> dict[str, object]:
    root = root.resolve()
    config = load_hybrid_difficulty_product_config(config_path, root=root)
    base = _chain(config.base_aggressive_manifest, kind="base")
    defensive_hard = _chain(config.defensive.hard_manifest, kind="hard")
    explorer_hard = _chain(config.explorer.hard_manifest, kind="hard")
    defensive_new = _chain(_resolve(root, defensive_extension) / "manifest.json", kind="extension")
    explorer_new = _chain(_resolve(root, explorer_extension) / "manifest.json", kind="extension")
    checks = {
        "base_aggressive_source": _base_source(config, base),
        "defensive_hard_source": _hard_source(
            config,
            defensive_hard,
            style="defensive",
        ),
        "explorer_hard_source": _hard_source(
            config,
            explorer_hard,
            style="explorer",
        ),
        "defensive_extension_source": _extension_source(
            config,
            defensive_new,
            style="defensive",
        ),
        "explorer_extension_source": _extension_source(
            config,
            explorer_new,
            style="explorer",
        ),
    }
    rows = list(base["rows"])
    rows.extend(
        {**row, "difficulty": "hard"}
        for row in defensive_hard["rows"]
        if row.get("policy") == "defensive"
    )
    rows.extend(
        {**row, "difficulty": "hard"}
        for row in explorer_hard["rows"]
        if row.get("policy") == "explorer"
    )
    rows.extend(defensive_new["rows"])
    rows.extend(explorer_new["rows"])
    matrix = evaluate_hybrid_all_style_matrix(
        rows,
        expected_pairs_per_opponent=10,
        expected_scenario_hash=config.scenario_hash,
        style_source_gates={
            "aggressive": checks["base_aggressive_source"],
            "defensive": checks["defensive_hard_source"]
            and checks["defensive_extension_source"],
            "explorer": checks["explorer_hard_source"]
            and checks["explorer_extension_source"],
        },
    )
    checks["matrix"] = bool(matrix["passed"])
    no_test_access = all(
        payload.get("test_cases_accessed") is False
        for chain in (
            base,
            defensive_hard,
            explorer_hard,
            defensive_new,
            explorer_new,
        )
        for payload in (chain["manifest"], chain["summary"])
    )
    checks["no_test_access"] = no_test_access
    passed = all(checks.values())
    checkpoint_hashes = base["summary"].get("checkpoint_sha256")
    if not isinstance(checkpoint_hashes, dict):
        checkpoint_hashes = {}
    return {
        **matrix,
        "schema_version": 1,
        "stage": "m5-hybrid-all-style-difficulty",
        "complete": passed,
        "passed": passed,
        "gates": {**checks, **matrix["gates"]},
        "policy_artifact_sha256": {
            "strong_base": checkpoint_hashes.get("strong_base"),
            "aggressive": checkpoint_hashes.get("aggressive"),
            "defensive": config.defensive.governor_config_sha256,
            "explorer": config.explorer.governor_config_sha256,
        },
        "policy_kinds": {
            "strong_base": "checkpoint",
            "aggressive": "checkpoint",
            "defensive": "hybrid_config",
            "explorer": "hybrid_config",
        },
        "source_manifest_sha256": {
            "base_aggressive": config.base_aggressive_manifest_sha256,
            "defensive_hard": config.defensive.hard_manifest_sha256,
            "defensive_extension": sha256_file(defensive_new["manifest_path"]),
            "explorer_hard": config.explorer.hard_manifest_sha256,
            "explorer_extension": sha256_file(explorer_new["manifest_path"]),
        },
        "config_sha256": config.config_sha256,
        "difficulty_config_sha256": config.difficulty_config_sha256,
        "cases_sha256": config.cases_sha256,
        "scenario_hash": config.scenario_hash,
        "test_cases_accessed": False,
    }


def evaluate_hybrid_all_style_matrix(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_pairs_per_opponent: int,
    expected_scenario_hash: str,
    style_source_gates: Mapping[str, bool],
) -> dict[str, object]:
    if (
        expected_pairs_per_opponent <= 0
        or set(style_source_gates) != {"aggressive", "defensive", "explorer"}
    ):
        raise ValueError("Hybrid all-style matrix settings are invalid")
    expected_per_cell = len(DUEL_OPPONENTS) * expected_pairs_per_opponent * 2
    expected = len(POLICIES) * len(DIFFICULTIES) * expected_per_cell
    identities = [_identity(row) for row in rows]
    cell_rows = {
        policy: {
            difficulty: [
                row
                for row in rows
                if row.get("policy") == policy
                and row.get("difficulty") == difficulty
            ]
            for difficulty in DIFFICULTIES
        }
        for policy in POLICIES
    }
    schedule_complete = (
        len(rows) == expected
        and len(identities) == len(set(identities))
        and all(
            _cell_complete(
                cell_rows[policy][difficulty],
                expected_per_cell=expected_per_cell,
                expected_pairs_per_opponent=expected_pairs_per_opponent,
            )
            for policy in POLICIES
            for difficulty in DIFFICULTIES
        )
    )
    protocol_clean = len(identities) == len(set(identities)) and all(
        row.get("scenario_hash") == expected_scenario_hash
        and row.get("terminated") is True
        and row.get("truncated") is False
        and row.get("protocol_inconsistent") is False
        and row.get("action_tic_inconsistent") is False
        and row.get("score_event_inconsistent") is False
        and row.get("peer_tic_lag_max") == 0
        for row in rows
    )
    case_identity = _shared_case_identity(rows)
    available = all(
        cell_rows[policy][difficulty]
        for policy in POLICIES
        for difficulty in DIFFICULTIES
    )
    cells = {
        policy: {
            difficulty: _cell(cell_rows[policy][difficulty])
            for difficulty in DIFFICULTIES
            if cell_rows[policy][difficulty]
        }
        for policy in POLICIES
    }
    aggregate_monotonic = available and all(
        _approximately_monotonic(
            [cells[policy][difficulty]["performance"] for difficulty in DIFFICULTIES]
        )
        for policy in POLICIES
    )
    monotonic_opponents = {
        policy: sum(
            _approximately_monotonic(
                [
                    _opponent_performance(
                        cell_rows[policy][difficulty],
                        opponent,
                    )
                    for difficulty in DIFFICULTIES
                ]
            )
            for opponent in DUEL_OPPONENTS
        )
        for policy in POLICIES
    } if available else {}
    objective_capability = available and all(
        cells[policy]["easy"]["objective_rate"] + 1e-12
        >= 0.70 * cells[policy]["hard"]["objective_rate"]
        and cells[policy]["normal"]["objective_rate"] + 1e-12
        >= 0.85 * cells[policy]["hard"]["objective_rate"]
        for policy in POLICIES
    )
    retention, retention_passed = _retention(cell_rows) if available else ({}, False)
    gates = {
        "complete": schedule_complete,
        "protocol_clean": protocol_clean,
        "shared_case_identity": case_identity,
        "aggregate_monotonic": aggregate_monotonic,
        "per_opponent_monotonic": (
            len(monotonic_opponents) == len(POLICIES)
            and all(value >= 4 for value in monotonic_opponents.values())
        ),
        "objective_capability": objective_capability,
        "hybrid_skill_retention": retention_passed,
        "style_sources_passed": all(style_source_gates.values()),
    }
    return {
        "complete": schedule_complete,
        "passed": all(gates.values()),
        "episodes": len(rows),
        "expected_episodes": expected,
        "protocol_inconsistencies": (
            sum(not _protocol_row(row, expected_scenario_hash) for row in rows)
            + len(identities)
            - len(set(identities))
        ),
        "monotonic_opponents": monotonic_opponents,
        "retention": retention,
        "cells": cells,
        "gates": gates,
        "test_cases_accessed": False,
    }


def _base_source(
    config: HybridDifficultyProductConfig,
    chain: Mapping[str, Any],
) -> bool:
    manifest = chain["manifest"]
    summary = chain["summary"]
    gates = summary.get("gates")
    return (
        manifest.get("passed") is True
        and manifest.get("episodes") == 600
        and manifest.get("expected_episodes") == 600
        and manifest.get("config_sha256") == config.difficulty_config_sha256
        and manifest.get("cases_sha256") == config.cases_sha256
        and manifest.get("scenario_hash") == config.scenario_hash
        and summary.get("stage") == "m5-difficulty"
        and summary.get("passed") is True
        and summary.get("complete") is True
        and isinstance(gates, dict)
        and bool(gates)
        and all(value is True for value in gates.values())
        and chain["hash_chain"]
    )


def _hard_source(
    config: HybridDifficultyProductConfig,
    chain: Mapping[str, Any],
    *,
    style: str,
) -> bool:
    source = config.defensive if style == "defensive" else config.explorer
    manifest = chain["manifest"]
    summary = chain["summary"]
    product = summary.get("product")
    product_gates = product.get("gates") if isinstance(product, dict) else None
    return (
        manifest.get("stage") == "m5-hybrid"
        and manifest.get("style") == style
        and manifest.get("passed") is True
        and manifest.get("episodes") == 200
        and manifest.get("expected_episodes") == 200
        and manifest.get("governor_config_sha256")
        == source.governor_config_sha256
        and manifest.get("cases_sha256") == config.cases_sha256
        and manifest.get("scenario_hash") == config.scenario_hash
        and summary.get("stage") == "m5-hybrid"
        and summary.get("style") == style
        and isinstance(product, dict)
        and product.get("passed") is True
        and isinstance(product_gates, dict)
        and bool(product_gates)
        and all(value is True for value in product_gates.values())
        and chain["hash_chain"]
    )


def _extension_source(
    config: HybridDifficultyProductConfig,
    chain: Mapping[str, Any],
    *,
    style: str,
) -> bool:
    source = config.defensive if style == "defensive" else config.explorer
    manifest = chain["manifest"]
    summary = chain["summary"]
    gates = summary.get("gates")
    return (
        manifest.get("stage") == f"m5-hybrid-{style}-difficulty-extension"
        and manifest.get("style") == style
        and manifest.get("passed") is True
        and manifest.get("episodes") == 200
        and manifest.get("expected_episodes") == 200
        and manifest.get("product_config_sha256") == config.config_sha256
        and manifest.get("governor_config_sha256")
        == source.governor_config_sha256
        and manifest.get("difficulty_config_sha256")
        == config.difficulty_config_sha256
        and manifest.get("cases_sha256") == config.cases_sha256
        and manifest.get("scenario_hash") == config.scenario_hash
        and summary.get("passed") is True
        and summary.get("complete") is True
        and isinstance(gates, dict)
        and bool(gates)
        and all(value is True for value in gates.values())
        and chain["hash_chain"]
    )


def _chain(manifest_path: Path, *, kind: str) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    directory = manifest_path.parent
    paths = {
        "manifest_path": manifest_path,
        "summary_path": directory / "summary.json",
        "episodes_path": directory / "episodes.jsonl",
    }
    if kind in ("hard", "extension"):
        paths["telemetry_path"] = directory / "telemetry.jsonl"
    if kind == "extension":
        paths["execution_trace_path"] = directory / "execution-trace.jsonl"
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing hybrid difficulty evidence: " + ", ".join(missing))
    manifest = _json(paths["manifest_path"])
    summary = _json(paths["summary_path"])
    hash_chain = (
        manifest.get("episodes_sha256") == sha256_file(paths["episodes_path"])
        and manifest.get("summary_sha256") == sha256_file(paths["summary_path"])
    )
    if "telemetry_path" in paths:
        hash_chain = hash_chain and manifest.get("telemetry_sha256") == sha256_file(
            paths["telemetry_path"]
        )
    if "execution_trace_path" in paths:
        hash_chain = hash_chain and manifest.get(
            "execution_trace_sha256"
        ) == sha256_file(paths["execution_trace_path"])
    return {
        **paths,
        "manifest": manifest,
        "summary": summary,
        "rows": _jsonl(paths["episodes_path"]),
        "hash_chain": hash_chain,
    }


def _cell(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "episodes": len(rows),
        "win_rate": sum(row.get("outcome") == "win" for row in rows) / len(rows),
        "objective_rate": sum(row.get("objective_completed") is True for row in rows)
        / len(rows),
        "mean_score_difference": float(
            np.mean(
                [
                    float(row["learner_score"]) - float(row["opponent_score"])
                    for row in rows
                ]
            )
        ),
        "performance": float(np.mean([float(row["performance"]) for row in rows])),
    }


def _cell_complete(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_per_cell: int,
    expected_pairs_per_opponent: int,
) -> bool:
    return len(rows) == expected_per_cell and Counter(
        row.get("opponent") for row in rows
    ) == Counter(
        {opponent: expected_pairs_per_opponent * 2 for opponent in DUEL_OPPONENTS}
    )


def _retention(
    cells: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
) -> tuple[dict[str, object], bool]:
    payload = {}
    passed = True
    for style in ("defensive", "explorer"):
        payload[style] = {}
        for difficulty in DIFFICULTIES:
            base = cells["strong_base"][difficulty]
            styled = cells[style][difficulty]
            base_performance = float(np.mean([float(row["performance"]) for row in base]))
            style_performance = float(
                np.mean([float(row["performance"]) for row in styled])
            )
            aggregate = style_performance / base_performance if base_performance > 0 else None
            opponents = {}
            for opponent in DUEL_OPPONENTS:
                base_value = _opponent_performance(base, opponent)
                style_value = _opponent_performance(styled, opponent)
                opponents[opponent] = (
                    style_value / base_value if base_value > 0 else None
                )
            cell_passed = (
                aggregate is not None
                and aggregate >= 0.85
                and all(value is not None and value >= 0.75 for value in opponents.values())
            )
            passed &= cell_passed
            payload[style][difficulty] = {
                "aggregate": aggregate,
                "per_opponent": opponents,
                "passed": cell_passed,
            }
    return payload, passed


def _shared_case_identity(rows: Sequence[Mapping[str, object]]) -> bool:
    by_case: dict[tuple[object, ...], set[tuple[object, ...]]] = {}
    for row in rows:
        key = (
            row.get("opponent"),
            row.get("pair_index"),
            row.get("learner_side"),
        )
        by_case.setdefault(key, set()).add(
            (row.get("seed"), row.get("scenario_hash"))
        )
    return len(by_case) == 100 and all(len(values) == 1 for values in by_case.values())


def _approximately_monotonic(values: Sequence[float]) -> bool:
    return all(
        left <= right + MONOTONIC_TOLERANCE
        for left, right in zip(values, values[1:], strict=False)
    )


def _opponent_performance(
    rows: Sequence[Mapping[str, object]],
    opponent: str,
) -> float:
    values = [
        float(row["performance"]) for row in rows if row.get("opponent") == opponent
    ]
    return float(np.mean(values)) if values else 0.0


def _identity(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row.get("policy"),
        row.get("difficulty"),
        row.get("opponent"),
        row.get("pair_index"),
        row.get("learner_side"),
    )


def _protocol_row(
    row: Mapping[str, object],
    scenario_hash: str,
) -> bool:
    return (
        row.get("scenario_hash") == scenario_hash
        and row.get("terminated") is True
        and row.get("truncated") is False
        and row.get("protocol_inconsistent") is False
        and row.get("action_tic_inconsistent") is False
        and row.get("score_event_inconsistent") is False
        and row.get("peer_tic_lag_max") == 0
    )


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects: {path}")
    return rows
