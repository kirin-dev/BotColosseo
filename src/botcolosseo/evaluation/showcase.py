from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
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
_METRIC_FIELDS = {
    "schema_version",
    "stage",
    "split",
    "passed",
    "style_gate_passed",
    "retention_gate_passed",
    "episodes",
    "checkpoint_sha256",
    "headline",
    "case_contrast_scores",
    "decision_contrast_scores",
}
_HEADLINE_FIELDS = {
    "base_win_rate",
    "aggressive_style_delta",
    "skill_retention",
}
_RECORD_ELIGIBILITY_FIELDS = {
    "terminated",
    "truncated",
    "objective_completed",
    "environment_attempts",
    "peer_tic_lag_max",
    "protocol_inconsistent",
    "action_tic_inconsistent",
    "score_event_inconsistent",
}
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


@dataclass(frozen=True)
class ShowcaseMetricEvidence:
    episodes: int
    base_win_rate: float
    aggressive_style_delta: float
    skill_retention: float
    checkpoint_sha256: dict[str, str]
    case_contrast_scores: dict[str, float]
    decision_contrast_scores: dict[str, tuple[float, ...]]
    source_sha256: str


@dataclass(frozen=True)
class ShowcaseSelection:
    selected_case_id: str
    selected_records: tuple[Mapping[str, object], ...]
    ranking: tuple[tuple[str, float], ...]
    rejection_reasons: dict[str, tuple[str, ...]]


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


def load_metric_evidence(
    path: Path,
    *,
    expected_stage: str,
    expected_hashes: Mapping[str, str],
) -> ShowcaseMetricEvidence:
    payload_bytes = path.read_bytes()
    payload = _load_json_object_from_bytes(payload_bytes, "Showcase metric evidence")
    _require_fields(payload, _METRIC_FIELDS, "Showcase metric evidence")
    if payload["schema_version"] != 1:
        raise ValueError("Showcase metric evidence requires schema_version 1")
    if payload["stage"] != expected_stage:
        raise ValueError("Showcase metric evidence stage does not match")
    if payload["split"] != "validation":
        raise ValueError("Showcase metric evidence must use the validation split")
    if any(
        payload[field] is not True
        for field in ("passed", "style_gate_passed", "retention_gate_passed")
    ):
        raise ValueError("Showcase metric evidence requires all gates to pass")

    episodes = payload["episodes"]
    if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes <= 0:
        raise ValueError("Showcase metric evidence requires positive episodes")

    checkpoint_sha256 = _load_hash_map(
        payload["checkpoint_sha256"], "Showcase metric checkpoint hashes"
    )
    expected_checkpoint_sha256 = _load_hash_map(
        expected_hashes, "Expected showcase checkpoint hashes"
    )
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("Showcase metric checkpoint hashes do not match")

    headline = payload["headline"]
    _require_fields(headline, _HEADLINE_FIELDS, "Showcase metric headline")
    base_win_rate = _finite_number(headline["base_win_rate"], "Base win rate")
    aggressive_style_delta = _finite_number(
        headline["aggressive_style_delta"], "Aggressive style delta"
    )
    skill_retention = _finite_number(headline["skill_retention"], "Skill retention")
    if not 0 <= base_win_rate <= 1 or not 0 <= skill_retention <= 1:
        raise ValueError("Showcase metric rates must be in [0, 1]")
    if aggressive_style_delta <= 0:
        raise ValueError("Showcase metric style delta must be positive")

    case_contrast_scores = _load_score_map(
        payload["case_contrast_scores"], "Showcase case contrast scores"
    )
    decision_contrast_scores = _load_decision_score_map(
        payload["decision_contrast_scores"], "Showcase decision contrast scores"
    )
    if set(case_contrast_scores) != set(decision_contrast_scores):
        raise ValueError("Showcase contrast score case keys do not match")
    return ShowcaseMetricEvidence(
        episodes=episodes,
        base_win_rate=base_win_rate,
        aggressive_style_delta=aggressive_style_delta,
        skill_retention=skill_retention,
        checkpoint_sha256=checkpoint_sha256,
        case_contrast_scores=case_contrast_scores,
        decision_contrast_scores=decision_contrast_scores,
        source_sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )


def select_showcase_case(
    records: Sequence[Mapping[str, object]],
    policy_ids: Sequence[str],
    contrast_scores: Mapping[str, float],
    *,
    require_normal_termination: bool = True,
) -> ShowcaseSelection:
    if not isinstance(require_normal_termination, bool):
        raise ValueError("Showcase termination requirement must be boolean")
    configured_policy_ids = tuple(policy_ids)
    if (
        not configured_policy_ids
        or any(
            not isinstance(policy_id, str) or not policy_id
            for policy_id in configured_policy_ids
        )
        or len(set(configured_policy_ids)) != len(configured_policy_ids)
    ):
        raise ValueError("Showcase policy IDs must be unique non-empty strings")
    validated_contrast_scores = _load_score_map(
        contrast_scores, "Showcase case contrast scores"
    )

    grouped: dict[str, dict[str, dict[str, object]]] = {}
    for record in records:
        case_id_value, policy_id = _validate_showcase_record(record, configured_policy_ids)
        by_policy = grouped.setdefault(case_id_value, {})
        if policy_id in by_policy:
            raise ValueError("Showcase records contain duplicate case and policy IDs")
        by_policy[policy_id] = dict(record)

    expected_policy_set = set(configured_policy_ids)
    eligible: list[tuple[str, float]] = []
    rejection_reasons: dict[str, tuple[str, ...]] = {}
    for case_id_value in sorted(grouped):
        by_policy = grouped[case_id_value]
        if set(by_policy) != expected_policy_set:
            rejection_reasons[case_id_value] = (
                "policy coverage does not match configured policies",
            )
            continue
        reasons = tuple(
            reason
            for policy_id in configured_policy_ids
            for reason in _ineligibility_reasons(
                by_policy[policy_id],
                require_normal_termination=require_normal_termination,
            )
        )
        if reasons:
            rejection_reasons[case_id_value] = reasons
            continue
        if case_id_value not in validated_contrast_scores:
            raise ValueError("Eligible showcase case is missing a contrast score")
        score = validated_contrast_scores[case_id_value]
        eligible.append((case_id_value, score))

    ranking = tuple(sorted(eligible, key=lambda item: (-item[1], item[0])))
    if not ranking:
        raise ValueError("No showcase case satisfies publication eligibility")
    selected_case_id = ranking[0][0]
    selected_records = tuple(
        dict(grouped[selected_case_id][policy_id]) for policy_id in configured_policy_ids
    )
    return ShowcaseSelection(
        selected_case_id=selected_case_id,
        selected_records=selected_records,
        ranking=ranking,
        rejection_reasons=rejection_reasons,
    )


def select_highlight_window(
    scores: Sequence[float], *, window_frames: int
) -> tuple[int, int]:
    if not scores or window_frames <= 0:
        raise ValueError("Highlight selection requires scores and a positive window")
    width = min(window_frames, len(scores))
    totals = [
        sum(scores[start : start + width])
        for start in range(len(scores) - width + 1)
    ]
    best = max(range(len(totals)), key=lambda index: (totals[index], -index))
    return best, best + width


def canonical_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            for row in rows:
                handle.write(canonical_json(dict(row)))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def build_showcase_manifest(
    *,
    git_commit: str,
    git_dirty: bool,
    config: ShowcaseConfig,
    scenario_hash: str,
    case_manifest_sha256: str,
    checkpoint_sha256: Mapping[str, str],
    metric_sha256: str | None,
    episodes_path: Path,
    selected_case: str,
    highlight: tuple[int, int],
    media: Sequence[Mapping[str, object]],
    gate_passed: bool,
) -> dict[str, object]:
    if not git_commit or not isinstance(git_dirty, bool):
        raise ValueError("Showcase manifest requires Git provenance")
    _validate_sha256(scenario_hash, "Showcase scenario hash")
    _validate_sha256(case_manifest_sha256, "Showcase case-manifest hash")
    checkpoints = _load_hash_map(
        checkpoint_sha256, "Showcase manifest checkpoint hashes"
    )
    if tuple(checkpoints) != tuple(policy.policy_id for policy in config.policies):
        raise ValueError("Showcase manifest checkpoint order does not match config")
    if config.publication:
        _validate_sha256(metric_sha256, "Showcase metric hash")
    elif metric_sha256 is not None or gate_passed:
        raise ValueError("Development showcase cannot claim metric or gate evidence")
    if not isinstance(gate_passed, bool):
        raise ValueError("Showcase gate state must be boolean")
    if not selected_case:
        raise ValueError("Showcase manifest requires a selected case")
    if (
        len(highlight) != 2
        or any(type(value) is not int for value in highlight)
        or highlight[0] < 0
        or highlight[1] <= highlight[0]
    ):
        raise ValueError("Showcase highlight range is invalid")
    media_rows = [_validate_media_row(row) for row in media]
    if not media_rows:
        raise ValueError("Showcase manifest requires media")

    identity_payload = {
        "config_sha256": config.config_sha256,
        "scenario_hash": scenario_hash,
        "case_manifest_sha256": case_manifest_sha256,
        "checkpoint_sha256": checkpoints,
        "metric_sha256": metric_sha256,
        "selected_case": selected_case,
        "highlight": list(highlight),
    }
    run_identity = hashlib.sha256(canonical_json(identity_payload)).hexdigest()
    root = config.config_path.parents[2]
    episode_target = config.evidence_dir / "episodes.jsonl"
    try:
        episode_log = episode_target.relative_to(root).as_posix()
    except ValueError:
        if config.publication:
            raise ValueError(
                "Public showcase evidence must stay inside the repository"
            ) from None
        episode_log = episode_target.name
    return {
        "schema_version": 1,
        "stage": config.stage,
        "publication": config.publication,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config_sha256": config.config_sha256,
        "scenario_hash": scenario_hash,
        "case_manifest_sha256": case_manifest_sha256,
        "checkpoint_sha256": checkpoints,
        "metric_sha256": metric_sha256,
        "policies": [
            {"id": policy.policy_id, "label": policy.label}
            for policy in config.policies
        ],
        "episode_log": episode_log,
        "episode_log_sha256": hashlib.sha256(episodes_path.read_bytes()).hexdigest(),
        "selected_case": selected_case,
        "highlight": list(highlight),
        "media": media_rows,
        "gate_identity": config.stage,
        "gate_passed": gate_passed,
        "split": "validation",
        "official_test_result": False,
        "test_cases_accessed": False,
        "run_identity": run_identity,
    }


def publish_staged_files(
    files: Iterable[tuple[Path, Path]],
    *,
    staged_manifest: Path,
    target_manifest: Path,
    run_identity: str,
) -> None:
    _validate_sha256(run_identity, "Showcase run identity")
    staged_payload = _load_json_object(staged_manifest, "Staged showcase manifest")
    if staged_payload.get("run_identity") != run_identity:
        raise ValueError("Staged showcase manifest identity does not match")
    if target_manifest.exists():
        target_payload = _load_json_object(target_manifest, "Published showcase manifest")
        if target_payload.get("run_identity") != run_identity:
            raise ValueError("Published showcase manifest identity does not match")

    transfers = tuple(files)
    if not transfers or any(not source.is_file() for source, _ in transfers):
        raise ValueError("Showcase publication requires every staged file")
    targets = [target.resolve() for _, target in transfers]
    if len(set(targets)) != len(targets) or target_manifest.resolve() in targets:
        raise ValueError("Showcase publication targets must be unique")
    for source, target in transfers:
        _copy_atomic(source, target)
    _copy_atomic(staged_manifest, target_manifest)


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


def _load_json_object_from_bytes(payload_bytes: bytes, description: str) -> dict[str, Any]:
    payload = json.loads(payload_bytes)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be an object")
    return payload


def _load_json_list(path: Path, description: str) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{description} must be a list")
    return payload


def _load_hash_map(payload: Any, description: str) -> dict[str, str]:
    if not isinstance(payload, Mapping) or not all(
        isinstance(key, str)
        and isinstance(value, str)
        and _SHA256.fullmatch(value) is not None
        for key, value in payload.items()
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256 map")
    return dict(payload)


def _finite_number(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be a finite number")
    return result


def _load_score_map(payload: Any, description: str) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{description} must be an object")
    scores: dict[str, float] = {}
    for case_id_value, score in payload.items():
        if not isinstance(case_id_value, str) or not case_id_value:
            raise ValueError(f"{description} requires non-empty case IDs")
        scores[case_id_value] = _finite_number(score, description)
    return scores


def _load_decision_score_map(
    payload: Any, description: str
) -> dict[str, tuple[float, ...]]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{description} must be an object")
    scores: dict[str, tuple[float, ...]] = {}
    for case_id_value, values in payload.items():
        if not isinstance(case_id_value, str) or not case_id_value:
            raise ValueError(f"{description} requires non-empty case IDs")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{description} requires non-empty score vectors")
        scores[case_id_value] = tuple(
            _finite_number(value, description) for value in values
        )
    return scores


def _validate_showcase_record(
    record: Mapping[str, object], policy_ids: Sequence[str]
) -> tuple[str, str]:
    if not isinstance(record, Mapping):
        raise ValueError("Showcase records must be mappings")
    case_id_value = record.get("case_id")
    policy_id = record.get("policy_id")
    if not isinstance(case_id_value, str) or not case_id_value:
        raise ValueError("Showcase record requires a non-empty case ID")
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("Showcase record requires a non-empty policy ID")
    if policy_id not in policy_ids:
        raise ValueError("Showcase record has an unknown policy ID")
    missing = _RECORD_ELIGIBILITY_FIELDS - set(record)
    if missing:
        raise ValueError("Showcase record is missing publication eligibility fields")
    if any(
        not isinstance(record[field], bool)
        for field in (
            "terminated",
            "truncated",
            "objective_completed",
            "protocol_inconsistent",
            "action_tic_inconsistent",
            "score_event_inconsistent",
        )
    ):
        raise ValueError("Showcase record eligibility flags must be booleans")
    if any(
        not isinstance(record[field], int) or isinstance(record[field], bool)
        for field in ("environment_attempts", "peer_tic_lag_max")
    ):
        raise ValueError("Showcase record attempt and tic fields must be integers")
    return case_id_value, policy_id


def _ineligibility_reasons(
    record: Mapping[str, object], *, require_normal_termination: bool
) -> tuple[str, ...]:
    termination_checks = (
        (record["terminated"] is not True, "episode did not terminate"),
        (record["truncated"] is not False, "episode was truncated"),
    ) if require_normal_termination else ()
    checks = (
        *termination_checks,
        (record["objective_completed"] is not True, "objective was not completed"),
        (record["environment_attempts"] != 1, "environment attempts were not one"),
        (record["peer_tic_lag_max"] != 0, "peer tic lag was nonzero"),
        (record["protocol_inconsistent"] is True, "protocol was inconsistent"),
        (record["action_tic_inconsistent"] is True, "action tic was inconsistent"),
        (record["score_event_inconsistent"] is True, "score event was inconsistent"),
    )
    return tuple(reason for failed, reason in checks if failed)


def _validate_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value


def _validate_media_row(row: Mapping[str, object]) -> dict[str, object]:
    expected = {"path", "sha256", "bytes", "frame_count", "dimensions", "fps"}
    if not isinstance(row, Mapping) or set(row) != expected:
        raise ValueError("Showcase media metadata has unknown or missing keys")
    path = row["path"]
    relative = PurePosixPath(path) if isinstance(path, str) else None
    if (
        relative is None
        or not path
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise ValueError("Showcase media path must be relative")
    _validate_sha256(row["sha256"], "Showcase media hash")
    if any(
        type(row[field]) is not int or row[field] <= 0
        for field in ("bytes", "frame_count", "fps")
    ):
        raise ValueError("Showcase media numeric metadata must be positive integers")
    dimensions = row["dimensions"]
    if (
        not isinstance(dimensions, list)
        or len(dimensions) != 2
        or any(type(value) is not int or value <= 0 for value in dimensions)
    ):
        raise ValueError("Showcase media dimensions must be two positive integers")
    return dict(row)


def _copy_atomic(source: Path, target: Path) -> None:
    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
