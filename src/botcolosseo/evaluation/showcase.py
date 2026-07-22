from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS, DuelCase

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONFIG_FIELDS = {
    "schema_version",
    "stage",
    "publication",
    "split",
    "cases",
    "metrics",
    "policies",
    "render",
    "evidence_dir",
}
_POLICY_FIELDS = {"id", "label", "checkpoint", "expected_sha256"}
_RENDER_FIELDS = {
    "fps",
    "gif_seconds",
    "gif_max_bytes",
    "max_decisions",
    "output_dir",
}
_CASE_FIELDS = {
    "split",
    "pair_index",
    "seed",
    "opponent",
    "learner_side",
    "core_spawn_index",
    "route",
}
_CASE_MANIFEST_FIELDS = {"schema_version", "source", "source_sha256", "cases"}
_EXPECTED_POLICY_IDS = {
    "development": ("ppo", "bc"),
    "m4": ("strong_base", "aggressive"),
    "m5": ("strong_base", "aggressive", "defensive", "explorer"),
}


@dataclass(frozen=True)
class ShowcasePolicySpec:
    policy_id: str
    label: str
    checkpoint: Path
    expected_sha256: str

    def __post_init__(self) -> None:
        if not self.policy_id or not self.label:
            raise ValueError("Showcase policy ID and label must be non-empty")
        if _SHA256.fullmatch(self.expected_sha256) is None:
            raise ValueError("Showcase policy requires a lowercase SHA-256")


@dataclass(frozen=True)
class ShowcaseRenderSpec:
    fps: int
    gif_seconds: int
    gif_max_bytes: int
    max_decisions: int

    def __post_init__(self) -> None:
        if not 0 < self.fps <= 30:
            raise ValueError("Showcase FPS must be in [1, 30]")
        if not 0 < self.gif_seconds <= 20:
            raise ValueError("Showcase GIF must be at most 20 seconds")
        if self.gif_max_bytes != 10_000_000:
            raise ValueError("Showcase GIF ceiling must remain 10000000 bytes")
        if self.max_decisions <= 0:
            raise ValueError("Showcase max decisions must be positive")


@dataclass(frozen=True)
class ShowcaseConfig:
    stage: str
    publication: bool
    split: str
    cases_path: Path
    metrics_path: Path | None
    policies: tuple[ShowcasePolicySpec, ...]
    render: ShowcaseRenderSpec
    output_dir: Path
    evidence_dir: Path
    config_path: Path
    config_sha256: str


def case_id(case: DuelCase) -> str:
    return f"{case.opponent}:{case.pair_index}:{case.learner_side}"


def load_showcase_config(path: Path, *, root: Path) -> ShowcaseConfig:
    root = root.resolve()
    config_path = _resolve_input_path(path, root)
    payload_bytes = config_path.read_bytes()
    payload = yaml.safe_load(payload_bytes)
    _require_fields(payload, _CONFIG_FIELDS, "Showcase config")
    if payload["schema_version"] != 1:
        raise ValueError("Showcase config requires schema_version 1")

    stage = payload["stage"]
    if not isinstance(stage, str):
        raise ValueError("Showcase stage must be a string")
    if stage not in _EXPECTED_POLICY_IDS:
        raise ValueError("Unsupported showcase stage")
    if payload["split"] != "validation":
        raise ValueError("Showcase configs must use the validation split")
    publication = payload["publication"]
    if not isinstance(publication, bool) or publication != (stage != "development"):
        raise ValueError("Showcase publication setting does not match stage")

    cases_path = _resolve_path(root, payload["cases"], "Showcase cases path")
    metrics = payload["metrics"]
    if (stage == "development") != (metrics is None):
        raise ValueError("Showcase metrics are required exactly for public stages")
    metrics_path = (
        None
        if metrics is None
        else _resolve_path(root, metrics, "Showcase metrics path")
    )

    policies = tuple(
        _load_policy(item, root=root) for item in _require_list(payload["policies"])
    )
    if tuple(policy.policy_id for policy in policies) != _EXPECTED_POLICY_IDS[stage]:
        raise ValueError("Showcase policies do not match the frozen stage order")

    render_payload = payload["render"]
    _require_fields(render_payload, _RENDER_FIELDS, "Showcase render config")
    render = ShowcaseRenderSpec(
        fps=render_payload["fps"],
        gif_seconds=render_payload["gif_seconds"],
        gif_max_bytes=render_payload["gif_max_bytes"],
        max_decisions=render_payload["max_decisions"],
    )
    output_dir = _resolve_path(root, render_payload["output_dir"], "Showcase output path")
    evidence_dir = _resolve_path(root, payload["evidence_dir"], "Showcase evidence path")
    _validate_output_roots(stage, root, output_dir, evidence_dir)
    return ShowcaseConfig(
        stage=stage,
        publication=publication,
        split=payload["split"],
        cases_path=cases_path,
        metrics_path=metrics_path,
        policies=policies,
        render=render,
        output_dir=output_dir,
        evidence_dir=evidence_dir,
        config_path=config_path,
        config_sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )


def load_showcase_cases(
    path: Path, *, root: Path, expected_count: int
) -> tuple[DuelCase, ...]:
    root = root.resolve()
    manifest_path = _resolve_input_path(path, root)
    manifest = _load_json_object(manifest_path, "Showcase case manifest")
    _require_fields(manifest, _CASE_MANIFEST_FIELDS, "Showcase case manifest")
    if manifest["schema_version"] != 1:
        raise ValueError("Showcase case manifest requires schema_version 1")
    if (
        not isinstance(manifest["source_sha256"], str)
        or _SHA256.fullmatch(manifest["source_sha256"]) is None
    ):
        raise ValueError("Showcase case manifest requires a lowercase source hash")

    source_path = _resolve_path(root, manifest["source"], "Showcase case source")
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("Showcase case source hash does not match")
    source_rows = _load_json_list(source_path, "Showcase case source")
    case_rows = _require_list(manifest["cases"])
    if len(case_rows) != expected_count:
        raise ValueError(f"Expected {expected_count} showcase cases, found {len(case_rows)}")

    cases: list[DuelCase] = []
    for row in case_rows:
        _require_fields(row, _CASE_FIELDS, "Showcase case")
        case = DuelCase(**row)
        if case.split != "validation":
            raise ValueError("Showcase cases must be validation-only")
        if case.opponent not in DUEL_OPPONENTS:
            raise ValueError("Showcase case has an unknown opponent")
        if not _case_exists_in_source(case, source_rows):
            raise ValueError("Showcase case does not match its source row")
        cases.append(case)
    if len({case_id(case) for case in cases}) != len(cases):
        raise ValueError("Showcase case manifest contains duplicate case IDs")
    return tuple(cases)


def _load_policy(payload: Any, *, root: Path) -> ShowcasePolicySpec:
    _require_fields(payload, _POLICY_FIELDS, "Showcase policy")
    return ShowcasePolicySpec(
        policy_id=payload["id"],
        label=payload["label"],
        checkpoint=_resolve_path(root, payload["checkpoint"], "Showcase checkpoint path"),
        expected_sha256=payload["expected_sha256"],
    )


def _validate_output_roots(
    stage: str, root: Path, output_dir: Path, evidence_dir: Path
) -> None:
    if stage == "development":
        expected_output = root / "artifacts/showcase-development/media"
        expected_evidence = root / "artifacts/showcase-development/evidence"
    else:
        expected_output = root / "docs/assets/showcase"
        expected_evidence = root / f"reports/showcase/{stage}"
    if output_dir != expected_output or evidence_dir != expected_evidence:
        raise ValueError("Showcase output and evidence roots are frozen by stage")


def _case_exists_in_source(case: DuelCase, source_rows: list[Any]) -> bool:
    expected = case.to_dict()
    for source_row in source_rows:
        if not isinstance(source_row, dict):
            continue
        if any(
            source_row.get(field) != expected[field]
            for field in _CASE_FIELDS - {"opponent"}
        ):
            continue
        if "opponent" not in source_row or source_row["opponent"] == case.opponent:
            return True
    return False


def _resolve_path(root: Path, value: Any, description: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{description} must be a relative path")
    relative = PurePosixPath(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{description} must be a relative path")
    return (root / relative).resolve()


def _resolve_input_path(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _require_fields(payload: Any, expected: set[str], description: str) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{description} has unknown or missing keys")


def _require_list(payload: Any) -> list[Any]:
    if not isinstance(payload, list):
        raise ValueError("Showcase list field must be a list")
    return payload


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be an object")
    return payload


def _load_json_list(path: Path, description: str) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{description} must be a list")
    return payload
