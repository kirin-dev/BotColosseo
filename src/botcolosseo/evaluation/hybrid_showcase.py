from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml

from botcolosseo.agents.league_opponents import sha256_file

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_POLICY_IDS = ("strong_base", "aggressive", "defensive", "explorer")
_POLICY_LABELS = (
    "Strong Base (learned)",
    "Aggressive (learned style)",
    "Defensive (hybrid governor)",
    "Explorer (hybrid governor)",
)
_TOP_FIELDS = {
    "schema_version",
    "stage",
    "publication",
    "split",
    "test_cases_accessed",
    "cases",
    "policies",
    "evidence",
    "render",
}
_CASES_FIELDS = {"manifest", "expected_sha256", "selected_case_id"}
_POLICY_FIELDS = {"id", "label", "kind", "artifact", "expected_sha256"}
_EVIDENCE_FIELDS = {"style", "summary", "expected_sha256"}
_RENDER_FIELDS = {
    "fps",
    "gif_seconds",
    "gif_max_bytes",
    "max_decisions",
    "output_dir",
    "evidence_dir",
}


@dataclass(frozen=True)
class HybridShowcasePolicy:
    policy_id: str
    label: str
    kind: Literal["checkpoint", "hybrid_config"]
    artifact: Path
    expected_sha256: str


@dataclass(frozen=True)
class HybridShowcaseEvidence:
    style: Literal["aggressive", "defensive", "explorer"]
    summary: Path
    expected_sha256: str


@dataclass(frozen=True)
class HybridShowcaseConfig:
    cases_manifest: Path
    cases_sha256: str
    selected_case_id: str
    policies: tuple[HybridShowcasePolicy, ...]
    evidence: tuple[HybridShowcaseEvidence, ...]
    fps: int
    gif_seconds: int
    gif_max_bytes: int
    max_decisions: int
    output_dir: Path
    evidence_dir: Path
    config_path: Path
    config_sha256: str


def select_hybrid_showcase_case(
    *,
    aggressive_records: Sequence[Mapping[str, object]],
    defensive_records: Sequence[Mapping[str, object]],
    explorer_records: Sequence[Mapping[str, object]],
    defensive_telemetry: Sequence[Mapping[str, object]],
    explorer_telemetry: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    def key(row: Mapping[str, object]) -> tuple[str, int, str]:
        opponent = row.get("opponent")
        pair_index = row.get("pair_index")
        learner_side = row.get("learner_side")
        if (
            not isinstance(opponent, str)
            or type(pair_index) is not int
            or learner_side not in ("host", "opponent")
        ):
            raise ValueError("Hybrid showcase source row has an invalid case key")
        return opponent, pair_index, learner_side

    def by_policy(
        rows: Sequence[Mapping[str, object]], policy: str
    ) -> dict[tuple[str, int, str], Mapping[str, object]]:
        selected = [row for row in rows if row.get("policy") == policy]
        result = {key(row): row for row in selected}
        if len(result) != len(selected):
            raise ValueError("Hybrid showcase source has duplicate policy cases")
        return result

    base = by_policy(aggressive_records, "strong_base")
    aggressive = by_policy(aggressive_records, "aggressive")
    defensive = by_policy(defensive_records, "defensive")
    explorer = by_policy(explorer_records, "explorer")
    defensive_counts = Counter(
        key(row) for row in defensive_telemetry if row.get("intervened") is True
    )
    explorer_changes = Counter(
        key(row)
        for row in explorer_telemetry
        if row.get("intervened") is True
        and row.get("final_action") != row.get("base_action")
    )
    explorer_modes: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    for row in explorer_telemetry:
        mode = row.get("route_mode")
        if mode in ("upper", "lower", "flank"):
            explorer_modes[key(row)].add(mode)

    candidates = []
    for case_key in set(base) & set(aggressive) & set(defensive) & set(explorer):
        episode_rows = (
            base[case_key],
            aggressive[case_key],
            defensive[case_key],
            explorer[case_key],
        )
        if not all(
            row.get("terminated") is True
            and row.get("truncated") is False
            and row.get("protocol_inconsistent") is False
            and row.get("objective_completed") is True
            for row in episode_rows
        ):
            continue
        base_engagement = base[case_key].get("engagement_initiations_per_100_decisions")
        style_engagement = aggressive[case_key].get(
            "engagement_initiations_per_100_decisions"
        )
        if (
            isinstance(base_engagement, bool)
            or not isinstance(base_engagement, (int, float))
            or isinstance(style_engagement, bool)
            or not isinstance(style_engagement, (int, float))
        ):
            raise ValueError("Hybrid showcase Aggressive engagement signal is invalid")
        aggressive_shift = float(style_engagement) - float(base_engagement)
        row = {
            "case_id": f"{case_key[0]}:{case_key[1]}:{case_key[2]}",
            "aggressive_shift": aggressive_shift,
            "defensive_interventions": defensive_counts[case_key],
            "explorer_action_changes": explorer_changes[case_key],
            "explorer_mode_count": len(explorer_modes[case_key]),
        }
        if (
            aggressive_shift > 0.0
            and defensive_counts[case_key] > 0
            and explorer_changes[case_key] > 0
            and len(explorer_modes[case_key]) >= 2
        ):
            candidates.append(row)
    if not candidates:
        raise ValueError("No validation case exposes all three style mechanisms")
    maxima = {
        field: max(float(row[field]) for row in candidates)
        for field in (
            "aggressive_shift",
            "defensive_interventions",
            "explorer_action_changes",
            "explorer_mode_count",
        )
    }
    for row in candidates:
        row["contrast_score"] = sum(
            float(row[field]) / maxima[field] for field in maxima
        )
    ranking = sorted(candidates, key=lambda row: (-float(row["contrast_score"]), row["case_id"]))
    return {
        "selected_case_id": ranking[0]["case_id"],
        "selection_rule": "equal-weight normalized mechanism contrast",
        "eligible_cases": len(ranking),
        "ranking": ranking,
        "test_cases_accessed": False,
    }


def load_hybrid_showcase_config(path: Path, *, root: Path) -> HybridShowcaseConfig:
    root = root.resolve()
    config_path = path.resolve() if path.is_absolute() else (root / path).resolve()
    payload_bytes = config_path.read_bytes()
    payload = yaml.safe_load(payload_bytes)
    _fields(payload, _TOP_FIELDS, "Hybrid showcase config")
    if (
        payload["schema_version"] != 1
        or payload["stage"] != "hybrid_product"
        or payload["publication"] is not True
        or payload["split"] != "validation"
        or payload["test_cases_accessed"] is not False
    ):
        raise ValueError("Hybrid showcase publication boundary is invalid")
    cases = payload["cases"]
    _fields(cases, _CASES_FIELDS, "Hybrid showcase cases")
    cases_path = _path(root, cases["manifest"], "case manifest")
    cases_sha256 = _hash(cases["expected_sha256"], "case manifest")
    _verify(cases_path, cases_sha256, "case manifest")
    selected_case_id = cases["selected_case_id"]
    if not isinstance(selected_case_id, str) or selected_case_id.count(":") != 2:
        raise ValueError("Hybrid showcase selected case ID is invalid")

    policy_rows = payload["policies"]
    if not isinstance(policy_rows, list) or len(policy_rows) != 4:
        raise ValueError("Hybrid showcase requires four policies")
    policies = []
    for row in policy_rows:
        _fields(row, _POLICY_FIELDS, "Hybrid showcase policy")
        policies.append(
            HybridShowcasePolicy(
                policy_id=row["id"],
                label=row["label"],
                kind=row["kind"],
                artifact=_path(root, row["artifact"], "policy artifact"),
                expected_sha256=_hash(row["expected_sha256"], "policy artifact"),
            )
        )
    if tuple(row.policy_id for row in policies) != _POLICY_IDS:
        raise ValueError("Hybrid showcase policy order is invalid")
    if tuple(row.label for row in policies) != _POLICY_LABELS:
        raise ValueError("Hybrid showcase labels must disclose learned and hybrid policies")
    if tuple(row.kind for row in policies) != (
        "checkpoint",
        "checkpoint",
        "hybrid_config",
        "hybrid_config",
    ):
        raise ValueError("Hybrid showcase policy kinds are invalid")
    for row in policies:
        _verify(row.artifact, row.expected_sha256, f"{row.policy_id} artifact")

    evidence_rows = payload["evidence"]
    if not isinstance(evidence_rows, list) or len(evidence_rows) != 3:
        raise ValueError("Hybrid showcase requires three style evidence sources")
    evidence = []
    for row in evidence_rows:
        _fields(row, _EVIDENCE_FIELDS, "Hybrid showcase evidence")
        evidence.append(
            HybridShowcaseEvidence(
                style=row["style"],
                summary=_path(root, row["summary"], "style summary"),
                expected_sha256=_hash(row["expected_sha256"], "style summary"),
            )
        )
    if tuple(row.style for row in evidence) != ("aggressive", "defensive", "explorer"):
        raise ValueError("Hybrid showcase style evidence order is invalid")
    for row in evidence:
        _verify(row.summary, row.expected_sha256, f"{row.style} evidence")

    render = payload["render"]
    _fields(render, _RENDER_FIELDS, "Hybrid showcase render")
    fps = render["fps"]
    gif_seconds = render["gif_seconds"]
    gif_max_bytes = render["gif_max_bytes"]
    max_decisions = render["max_decisions"]
    if (
        type(fps) is not int
        or not 0 < fps <= 30
        or type(gif_seconds) is not int
        or not 0 < gif_seconds <= 20
        or gif_max_bytes != 10_000_000
        or type(max_decisions) is not int
        or max_decisions <= 0
    ):
        raise ValueError("Hybrid showcase render settings are invalid")
    output_dir = _path(root, render["output_dir"], "output directory")
    evidence_dir = _path(root, render["evidence_dir"], "evidence directory")
    if output_dir != root / "docs/assets/showcase":
        raise ValueError("Hybrid showcase output directory is not publication-safe")
    if evidence_dir != root / "reports/showcase/hybrid-product":
        raise ValueError("Hybrid showcase evidence directory is not frozen")
    return HybridShowcaseConfig(
        cases_manifest=cases_path,
        cases_sha256=cases_sha256,
        selected_case_id=selected_case_id,
        policies=tuple(policies),
        evidence=tuple(evidence),
        fps=fps,
        gif_seconds=gif_seconds,
        gif_max_bytes=gif_max_bytes,
        max_decisions=max_decisions,
        output_dir=output_dir,
        evidence_dir=evidence_dir,
        config_path=config_path,
        config_sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )


def _fields(value: object, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} has unknown or missing fields")


def _path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"Hybrid showcase {label} must be a relative path")
    relative = PurePosixPath(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Hybrid showcase {label} must be repository-relative")
    return (root / relative).resolve()


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"Hybrid showcase {label} hash is invalid")
    return value


def _verify(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != expected:
        raise ValueError(f"Hybrid showcase {label} hash does not match")
