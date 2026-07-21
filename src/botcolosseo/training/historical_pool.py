from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path, PurePosixPath

from botcolosseo.agents.league_opponents import sha256_file

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_POLICY_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_POOL_FIELDS = {
    "schema_version",
    "pool_version",
    "parent_manifest_sha256",
    "created_at_utc",
    "entries",
    "manifest_sha256",
}
_ENTRY_FIELDS = {
    "policy_id",
    "checkpoint",
    "checkpoint_sha256",
    "scenario_hash",
    "config_hash",
    "source_git_commit",
    "parent_checkpoint_sha256",
    "environment_steps",
    "admitted_at_utc",
    "validation_report",
    "validation_report_sha256",
    "script_average_win_rate",
    "script_worst_case_win_rate",
    "objective_rate",
    "payoff_by_policy",
    "anchor",
    "admission_reason",
}


def _valid_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


@dataclass(frozen=True)
class PoolEntry:
    policy_id: str
    checkpoint: str
    checkpoint_sha256: str
    scenario_hash: str
    config_hash: str
    source_git_commit: str
    parent_checkpoint_sha256: str
    environment_steps: int
    admitted_at_utc: str
    validation_report: str
    validation_report_sha256: str
    script_average_win_rate: float
    script_worst_case_win_rate: float
    objective_rate: float
    payoff_by_policy: dict[str, float]
    anchor: bool
    admission_reason: str

    def __post_init__(self) -> None:
        if _POLICY_ID_PATTERN.fullmatch(self.policy_id) is None:
            raise ValueError("Invalid pool policy ID")
        if not _valid_relative_path(self.checkpoint):
            raise ValueError("Pool checkpoint path must be relative")
        if not _valid_relative_path(self.validation_report):
            raise ValueError("Pool validation report path must be relative")
        if "test" in PurePosixPath(self.validation_report).name.lower():
            raise ValueError("Pool admission evidence must not be test-derived")
        for name, value in (
            ("checkpoint", self.checkpoint_sha256),
            ("parent checkpoint", self.parent_checkpoint_sha256),
            ("validation report", self.validation_report_sha256),
        ):
            if _SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(f"Invalid {name} SHA-256")
        if not self.scenario_hash or not self.config_hash or not self.source_git_commit:
            raise ValueError("Pool provenance fields must be non-empty")
        if self.environment_steps < 0 or not self.admitted_at_utc:
            raise ValueError("Invalid pool admission counters")
        rates = (
            self.script_average_win_rate,
            self.script_worst_case_win_rate,
            self.objective_rate,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("Pool rates must be finite probabilities")
        if not self.payoff_by_policy or any(
            not key
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            for key, value in self.payoff_by_policy.items()
        ):
            raise ValueError("Pool payoff vectors must contain finite probabilities")
        if not self.admission_reason:
            raise ValueError("Pool admission reason must be non-empty")


@dataclass(frozen=True)
class HistoricalPoolManifest:
    schema_version: int
    pool_version: int
    parent_manifest_sha256: str | None
    created_at_utc: str
    entries: tuple[PoolEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported historical pool schema version")
        if self.pool_version < 0 or not self.created_at_utc:
            raise ValueError("Invalid historical pool version")
        if not 1 <= len(self.entries) <= 12:
            raise ValueError("Historical pool must contain 1 to 12 entries")
        if self.pool_version == 0 and self.parent_manifest_sha256 is not None:
            raise ValueError("Initial pool cannot have a parent manifest")
        if self.pool_version > 0 and (
            self.parent_manifest_sha256 is None
            or _SHA256_PATTERN.fullmatch(self.parent_manifest_sha256) is None
        ):
            raise ValueError("Updated pool requires a parent manifest hash")
        policy_ids = [entry.policy_id for entry in self.entries]
        checkpoint_hashes = [entry.checkpoint_sha256 for entry in self.entries]
        if len(set(policy_ids)) != len(policy_ids):
            raise ValueError("Historical pool has duplicate policy IDs")
        if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
            raise ValueError("Historical pool has duplicate checkpoint hashes")
        if sum(entry.anchor for entry in self.entries) != 1 or not self.entries[0].anchor:
            raise ValueError("Historical pool requires one permanent first anchor")
        if len({entry.scenario_hash for entry in self.entries}) != 1:
            raise ValueError("Historical pool scenario hashes do not match")

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(_pool_payload(self))).hexdigest()


@dataclass(frozen=True)
class AdmissionMetrics:
    integrity_ok: bool
    validation_complete: bool
    paired_side_swapped: bool
    protocol_inconsistencies: int
    source_split: str
    candidate_script_average: float
    active_script_average: float
    candidate_historical_worst_case: float
    active_historical_worst_case: float
    candidate_payoffs: dict[str, float]
    active_payoffs: dict[str, dict[str, float]]


@dataclass(frozen=True)
class AdmissionDecision:
    eligible: bool
    reason: str
    replacement_policy_id: str | None = None


def _pool_payload(pool: HistoricalPoolManifest) -> dict[str, object]:
    return {
        "schema_version": pool.schema_version,
        "pool_version": pool.pool_version,
        "parent_manifest_sha256": pool.parent_manifest_sha256,
        "created_at_utc": pool.created_at_utc,
        "entries": [asdict(entry) for entry in pool.entries],
    }


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def write_pool_atomic(pool: HistoricalPoolManifest, path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**_pool_payload(pool), "manifest_sha256": pool.manifest_sha256}
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def load_pool(
    path: Path, *, verify_artifacts: bool = True, artifact_root: Path | None = None
) -> HistoricalPoolManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != _POOL_FIELDS:
        raise ValueError("Historical pool manifest fields do not match schema")
    entries_payload = payload["entries"]
    if not isinstance(entries_payload, list):
        raise ValueError("Historical pool entries must be a list")
    entries: list[PoolEntry] = []
    for row in entries_payload:
        if not isinstance(row, dict) or set(row) != _ENTRY_FIELDS:
            raise ValueError("Historical pool entry fields do not match schema")
        entries.append(PoolEntry(**row))
    pool = HistoricalPoolManifest(
        schema_version=payload["schema_version"],
        pool_version=payload["pool_version"],
        parent_manifest_sha256=payload["parent_manifest_sha256"],
        created_at_utc=payload["created_at_utc"],
        entries=tuple(entries),
    )
    if payload["manifest_sha256"] != pool.manifest_sha256:
        raise ValueError("Historical pool manifest hash does not match")
    if verify_artifacts:
        root = (artifact_root or Path.cwd()).expanduser().resolve()
        for entry in pool.entries:
            checkpoint = root / entry.checkpoint
            report = root / entry.validation_report
            if not checkpoint.is_file() or sha256_file(checkpoint) != entry.checkpoint_sha256:
                raise ValueError(f"{entry.policy_id} checkpoint hash does not match")
            if not report.is_file() or sha256_file(report) != entry.validation_report_sha256:
                raise ValueError(f"{entry.policy_id} validation report hash does not match")
    return pool


def _finite_probability(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= value <= 1.0


def _l1(left: dict[str, float], right: dict[str, float]) -> float:
    if set(left) != set(right):
        raise ValueError("Payoff vectors do not share the same coordinates")
    return sum(abs(left[key] - right[key]) for key in sorted(left))


def _minimum_pairwise(vectors: list[dict[str, float]]) -> float:
    distances = [_l1(left, right) for left, right in combinations(vectors, 2)]
    return min(distances) if distances else math.inf


def admission_decision(
    pool: HistoricalPoolManifest, entry: PoolEntry, metrics: AdmissionMetrics
) -> AdmissionDecision:
    if entry.policy_id in {active.policy_id for active in pool.entries}:
        return AdmissionDecision(False, "duplicate policy ID")
    if entry.checkpoint_sha256 in {
        active.checkpoint_sha256 for active in pool.entries
    }:
        return AdmissionDecision(False, "duplicate checkpoint hash")
    if entry.scenario_hash != pool.entries[0].scenario_hash:
        return AdmissionDecision(False, "scenario mismatch")
    if entry.payoff_by_policy != metrics.candidate_payoffs:
        return AdmissionDecision(False, "candidate payoff evidence does not match entry")
    if not metrics.integrity_ok:
        return AdmissionDecision(False, "integrity audit failed")
    if not metrics.validation_complete:
        return AdmissionDecision(False, "validation rows are incomplete")
    if not metrics.paired_side_swapped:
        return AdmissionDecision(False, "validation rows are not paired")
    if metrics.protocol_inconsistencies != 0:
        return AdmissionDecision(False, "protocol inconsistencies are nonzero")
    if metrics.source_split != "validation":
        return AdmissionDecision(False, "admission requires validation evidence")
    scalar_metrics = (
        metrics.candidate_script_average,
        metrics.active_script_average,
        metrics.candidate_historical_worst_case,
        metrics.active_historical_worst_case,
    )
    if any(not _finite_probability(value) for value in scalar_metrics):
        return AdmissionDecision(False, "validation metrics are not finite probabilities")
    if metrics.candidate_script_average < metrics.active_script_average - 0.10:
        return AdmissionDecision(False, "candidate regresses on the script pool")
    if not metrics.candidate_payoffs or any(
        not _finite_probability(value) for value in metrics.candidate_payoffs.values()
    ):
        return AdmissionDecision(False, "candidate payoff vector is incomplete")
    try:
        distances = {
            active.policy_id: _l1(
                metrics.candidate_payoffs, metrics.active_payoffs[active.policy_id]
            )
            for active in pool.entries
            if not active.anchor
        }
    except (KeyError, ValueError):
        return AdmissionDecision(False, "active payoff vectors are incomplete")
    worst_case_improved = (
        metrics.candidate_historical_worst_case
        > metrics.active_historical_worst_case
    )
    diverse = all(distance >= 0.10 for distance in distances.values())
    if not worst_case_improved and not diverse:
        return AdmissionDecision(False, "candidate has insufficient payoff diversity")
    if len(pool.entries) < 12:
        return AdmissionDecision(True, "eligible candidate admitted")

    protected = {pool.entries[0].policy_id, pool.entries[-1].policy_id}
    replaceable = [
        active for active in pool.entries if active.policy_id not in protected
    ]
    try:
        current_vectors = [metrics.active_payoffs[active.policy_id] for active in pool.entries]
        current_minimum = _minimum_pairwise(current_vectors)
        options: list[tuple[float, int, str]] = []
        for active in replaceable:
            vectors = [
                metrics.active_payoffs[other.policy_id]
                for other in pool.entries
                if other.policy_id != active.policy_id
            ] + [metrics.candidate_payoffs]
            options.append(
                (
                    _minimum_pairwise(vectors),
                    active.environment_steps,
                    active.policy_id,
                )
            )
    except (KeyError, ValueError):
        return AdmissionDecision(False, "active payoff vectors are incomplete")
    best_minimum = max(option[0] for option in options)
    eligible_options = [option for option in options if option[0] == best_minimum]
    replacement = min(eligible_options, key=lambda option: (option[1], option[2]))[2]
    if not worst_case_improved and best_minimum <= current_minimum:
        return AdmissionDecision(False, "capacity replacement does not improve diversity")
    return AdmissionDecision(True, "eligible candidate replaces redundant policy", replacement)


def admit_candidate(
    pool: HistoricalPoolManifest, entry: PoolEntry, metrics: AdmissionMetrics
) -> HistoricalPoolManifest:
    decision = admission_decision(pool, entry, metrics)
    if not decision.eligible:
        raise ValueError(decision.reason)
    entries = tuple(
        active
        for active in pool.entries
        if active.policy_id != decision.replacement_policy_id
    ) + (entry,)
    return HistoricalPoolManifest(
        schema_version=1,
        pool_version=pool.pool_version + 1,
        parent_manifest_sha256=pool.manifest_sha256,
        created_at_utc=entry.admitted_at_utc,
        entries=entries,
    )
