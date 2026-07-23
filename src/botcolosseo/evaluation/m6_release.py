from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

M6_POLICIES = ("strong_base", "aggressive", "defensive", "explorer")
_UPSTREAMS = ("m4", "defensive", "explorer", "difficulty")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def build_m6_showcase_metric_payload(
    *,
    m4: Mapping[str, object],
    m4_showcase: Mapping[str, object],
    defensive: Mapping[str, object],
    explorer: Mapping[str, object],
    difficulty: Mapping[str, object],
    upstream_sha256: Mapping[str, str],
) -> dict[str, object]:
    _require_passed(m4, stage="m4")
    _require_passed(defensive, stage="m5-defensive")
    _require_passed(explorer, stage="m5-explorer")
    _require_passed(difficulty, stage="m5-difficulty")
    upstream = _hash_map(upstream_sha256, "M6 upstream hashes")
    if tuple(upstream) != _UPSTREAMS:
        raise ValueError("M6 upstream evidence order is incomplete")

    m4_hashes = _hashes(m4)
    defensive_hashes = _hashes(defensive)
    explorer_hashes = _hashes(explorer)
    difficulty_hashes = _hashes(difficulty)
    if tuple(m4_hashes) != ("strong_base", "aggressive"):
        raise ValueError("M6 requires the frozen M4 policy identities")
    checkpoints = {
        "strong_base": m4_hashes["strong_base"],
        "aggressive": m4_hashes["aggressive"],
        "defensive": defensive_hashes.get("defensive", ""),
        "explorer": explorer_hashes.get("explorer", ""),
    }
    checkpoints = _hash_map(checkpoints, "M6 checkpoint hashes")
    if (
        defensive_hashes.get("strong_base") != checkpoints["strong_base"]
        or explorer_hashes.get("strong_base") != checkpoints["strong_base"]
        or difficulty_hashes != checkpoints
    ):
        raise ValueError("M6 upstream checkpoint identities are inconsistent")

    showcase_hashes = _hash_map(
        m4_showcase.get("checkpoint_sha256"), "M4 showcase checkpoint hashes"
    )
    if (
        m4_showcase.get("schema_version") != 1
        or m4_showcase.get("stage") != "m4"
        or m4_showcase.get("split") != "validation"
        or m4_showcase.get("passed") is not True
        or showcase_hashes != m4_hashes
    ):
        raise ValueError("M6 requires the passing hash-bound M4 showcase metrics")
    case_scores = _score_map(
        m4_showcase.get("case_contrast_scores"), "M4 case contrast scores"
    )
    decision_scores = _decision_score_map(
        m4_showcase.get("decision_contrast_scores"),
        "M4 decision contrast scores",
    )
    if set(case_scores) != set(decision_scores):
        raise ValueError("M4 showcase contrast identities are inconsistent")

    m4_retention = _rate(m4, "skill_retention")
    defensive_retention = _rate(defensive, "skill_retention")
    explorer_retention = _rate(explorer, "skill_retention")
    base_win_rate = _policy_rate(m4, "strong_base", "win_rate")
    aggressive_delta = _positive(m4, "engagement_initiation_delta")
    defensive_delta = _positive(defensive, "protective_presence_delta")
    explorer_delta = _positive(explorer, "route_entropy_delta")
    episode_total = sum(
        _positive_integer(payload, "episodes")
        for payload in (m4, defensive, explorer, difficulty)
    )
    minimum_retention = min(
        m4_retention,
        defensive_retention,
        explorer_retention,
    )
    return {
        "schema_version": 2,
        "stage": "m6",
        "split": "validation",
        "passed": True,
        "style_gate_passed": True,
        "retention_gate_passed": True,
        "difficulty_gate_passed": True,
        "episodes": episode_total,
        "checkpoint_sha256": checkpoints,
        "headline_cards": [
            {"label": "Base win rate", "value": f"{base_win_rate:.1%}"},
            {"label": "Aggressive shift", "value": f"{aggressive_delta:+.3f}"},
            {"label": "Defensive shift", "value": f"{defensive_delta:+.3f}"},
            {"label": "Explorer shift", "value": f"{explorer_delta:+.3f}"},
            {"label": "Min retention", "value": f"{minimum_retention:.1%}"},
            {"label": "Evidence", "value": f"{episode_total:,} eps"},
        ],
        "case_contrast_scores": case_scores,
        "decision_contrast_scores": decision_scores,
        "upstream_sha256": upstream,
        "test_cases_accessed": False,
    }


def _require_passed(payload: Mapping[str, object], *, stage: str) -> None:
    gates = payload.get("gates")
    if (
        payload.get("stage") != stage
        or payload.get("split") != "validation"
        or payload.get("passed") is not True
        or payload.get("complete") is not True
        or payload.get("test_cases_accessed") is not False
        or not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
    ):
        raise ValueError(f"M6 requires passing complete {stage} evidence")


def _hashes(payload: Mapping[str, object]) -> dict[str, str]:
    return _hash_map(payload.get("checkpoint_sha256"), "M6 source hashes")


def _hash_map(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} are invalid")
    result: dict[str, str] = {}
    for key, digest in value.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise ValueError(f"{label} are invalid")
        result[key] = digest
    return result


def _rate(payload: Mapping[str, object], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"M6 source {field} is invalid")
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError(f"M6 source {field} must be in [0, 1]")
    return result


def _positive(payload: Mapping[str, object], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"M6 source {field} is invalid")
    result = float(value)
    if not result > 0:
        raise ValueError(f"M6 source {field} must be positive")
    return result


def _positive_integer(payload: Mapping[str, object], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"M6 source {field} must be a positive integer")
    return value


def _policy_rate(
    payload: Mapping[str, object],
    policy: str,
    field: str,
) -> float:
    policies = payload.get("policies")
    if not isinstance(policies, Mapping) or not isinstance(
        policies.get(policy), Mapping
    ):
        raise ValueError("M6 source policy summary is missing")
    return _rate(policies[policy], field)  # type: ignore[arg-type]


def _score_map(value: object, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} are invalid")
    result: dict[str, float] = {}
    for key, score in value.items():
        if (
            not isinstance(key, str)
            or not key
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError(f"{label} are invalid")
        result[key] = float(score)
    return result


def _decision_score_map(value: object, label: str) -> dict[str, list[float]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} are invalid")
    result: dict[str, list[float]] = {}
    for key, scores in value.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(scores, Sequence)
            or isinstance(scores, (str, bytes))
            or not scores
        ):
            raise ValueError(f"{label} are invalid")
        parsed: list[float] = []
        for score in scores:
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise ValueError(f"{label} are invalid")
            parsed.append(float(score))
        result[key] = parsed
    return result
