from __future__ import annotations

import re
from collections.abc import Mapping

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_POLICIES = ("strong_base", "aggressive", "defensive", "explorer")


def build_hybrid_m6_metric_payload(
    *,
    aggressive: Mapping[str, object],
    defensive: Mapping[str, object],
    explorer: Mapping[str, object],
    difficulty: Mapping[str, object],
    showcase: Mapping[str, object],
    upstream_sha256: Mapping[str, str],
) -> dict[str, object]:
    _passed(aggressive, stage="m4")
    defensive_product = _hybrid(defensive, style="defensive")
    explorer_product = _hybrid(explorer, style="explorer")
    _passed(difficulty, stage="m5-hybrid-all-style-difficulty")
    if difficulty.get("episodes") != 1200:
        raise ValueError("Hybrid M6 metrics require the 1,200-episode matrix")
    artifacts = _hash_map(
        difficulty.get("policy_artifact_sha256"),
        "Hybrid M6 policy artifacts",
    )
    if set(artifacts) != set(_POLICIES):
        raise ValueError("Hybrid M6 policy artifact set is invalid")
    showcase_artifacts = _hash_map(
        showcase.get("policy_artifact_sha256"),
        "Hybrid M6 showcase artifacts",
    )
    if (
        showcase.get("stage") != "hybrid_product_showcase"
        or showcase.get("publication") is not True
        or showcase.get("test_cases_accessed") is not False
        or showcase_artifacts != artifacts
        or showcase.get("scenario_hash") != difficulty.get("scenario_hash")
    ):
        raise ValueError("Hybrid M6 showcase identity is invalid")
    upstream = _hash_map(upstream_sha256, "Hybrid M6 upstream hashes")
    if tuple(upstream) != (
        "aggressive",
        "defensive",
        "explorer",
        "difficulty",
        "showcase",
    ):
        raise ValueError("Hybrid M6 upstream evidence order is invalid")
    aggressive_hashes = _hash_map(
        aggressive.get("checkpoint_sha256"),
        "Aggressive checkpoint hashes",
    )
    if (
        aggressive_hashes.get("strong_base") != artifacts["strong_base"]
        or aggressive_hashes.get("aggressive") != artifacts["aggressive"]
    ):
        raise ValueError("Hybrid M6 learned checkpoint identity drifted")
    retention = difficulty.get("retention")
    if not isinstance(retention, Mapping):
        raise ValueError("Hybrid M6 matrix retention is missing")
    minimum_matrix_retention = _minimum_retention(retention)
    base_win_rate = _policy_rate(aggressive, "strong_base", "win_rate")
    aggressive_shift = _number(aggressive, "engagement_initiation_delta")
    defensive_retention = _number(defensive_product, "skill_retention")
    explorer_signature = _number(
        explorer_product,
        "route_action_signature_distance",
    )
    return {
        "schema_version": 1,
        "stage": "m6-hybrid-product-metrics",
        "split": "validation",
        "passed": True,
        "style_gate_passed": True,
        "retention_gate_passed": True,
        "difficulty_gate_passed": True,
        "anonymous_user_study_required": True,
        "showcase_ready": False,
        "episodes": 1200,
        "policy_artifact_sha256": artifacts,
        "policy_kinds": difficulty.get("policy_kinds"),
        "headline_cards": [
            {"label": "Base win rate", "value": f"{base_win_rate:.1%}"},
            {"label": "Aggressive shift", "value": f"{aggressive_shift:+.3f}"},
            {
                "label": "Defensive retention",
                "value": f"{defensive_retention:.1%}",
            },
            {
                "label": "Explorer signature",
                "value": f"{explorer_signature:.3f}",
            },
            {
                "label": "Min matrix retention",
                "value": f"{minimum_matrix_retention:.1%}",
            },
            {"label": "Difficulty matrix", "value": "1,200 eps"},
        ],
        "upstream_sha256": upstream,
        "scenario_hash": difficulty.get("scenario_hash"),
        "test_cases_accessed": False,
    }


def _passed(payload: Mapping[str, object], *, stage: str) -> None:
    gates = payload.get("gates")
    if (
        payload.get("stage") != stage
        or payload.get("passed") is not True
        or payload.get("complete") is not True
        or payload.get("test_cases_accessed") is not False
        or not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
    ):
        raise ValueError(f"Hybrid M6 requires passing {stage} evidence")


def _hybrid(
    payload: Mapping[str, object],
    *,
    style: str,
) -> Mapping[str, object]:
    product = payload.get("product")
    product_gates = product.get("gates") if isinstance(product, Mapping) else None
    if (
        payload.get("stage") != "m5-hybrid"
        or payload.get("style") != style
        or payload.get("test_cases_accessed") is not False
        or not isinstance(product, Mapping)
        or product.get("passed") is not True
        or product.get("complete") is not True
        or not isinstance(product_gates, Mapping)
        or not product_gates
        or any(value is not True for value in product_gates.values())
    ):
        raise ValueError(f"Hybrid M6 requires passing {style} product evidence")
    return product


def _minimum_retention(payload: Mapping[str, object]) -> float:
    values = []
    for style in ("defensive", "explorer"):
        tiers = payload.get(style)
        if not isinstance(tiers, Mapping):
            raise ValueError("Hybrid M6 matrix retention is invalid")
        for difficulty in ("easy", "normal", "hard"):
            cell = tiers.get(difficulty)
            if not isinstance(cell, Mapping) or cell.get("passed") is not True:
                raise ValueError("Hybrid M6 matrix retention is invalid")
            aggregate = cell.get("aggregate")
            opponents = cell.get("per_opponent")
            if not isinstance(opponents, Mapping):
                raise ValueError("Hybrid M6 matrix retention is invalid")
            values.append(_rate(aggregate, "matrix retention"))
            values.extend(
                _rate(value, "matrix opponent retention")
                for value in opponents.values()
            )
    return min(values)


def _policy_rate(
    payload: Mapping[str, object],
    policy: str,
    field: str,
) -> float:
    policies = payload.get("policies")
    if not isinstance(policies, Mapping):
        raise ValueError("Hybrid M6 policy metrics are missing")
    row = policies.get(policy)
    if not isinstance(row, Mapping):
        raise ValueError("Hybrid M6 policy metrics are missing")
    return _rate(row.get(field), field)


def _number(payload: Mapping[str, object], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Hybrid M6 {field} is invalid")
    return float(value)


def _rate(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Hybrid M6 {label} is invalid")
    result = float(value)
    if result < 0:
        raise ValueError(f"Hybrid M6 {label} is invalid")
    return result


def _hash_map(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} are invalid")
    result = {}
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
