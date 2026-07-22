from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from botcolosseo.agents.league_opponents import sha256_file

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_POLICY_ID = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_FORBIDDEN_SPLIT = re.compile(
    r"(?:^|[-_.])(test|heldout|held-out|held_out)(?:[-_.]|$)"
)
_EVIDENCE_FIELDS = {
    "schema_version",
    "policy_id",
    "checkpoint",
    "checkpoint_sha256",
    "environment_steps",
    "split",
    "test_cases_accessed",
    "integrity_passed",
    "rejection_reasons",
    "historical_worst_case_win_rate",
    "script_average_win_rate",
    "full_objective_rate",
    "config_hash",
    "pool_manifest_sha256",
    "payoff_report_sha256",
    "scenario_hash",
}
SELECTION_RULE = (
    "integrity_passed_desc",
    "historical_worst_case_win_rate_desc",
    "script_average_win_rate_desc",
    "full_objective_rate_desc",
    "environment_steps_asc",
)


def _relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _validation_only_path(path: PurePosixPath) -> bool:
    return not any(_FORBIDDEN_SPLIT.search(part.lower()) for part in path.parts)


def _probability(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= value <= 1.0


@dataclass(frozen=True)
class StrongBaseCandidate:
    policy_id: str
    checkpoint: str
    checkpoint_sha256: str
    validation_report: str
    validation_report_sha256: str
    environment_steps: int
    integrity_passed: bool
    rejection_reasons: tuple[str, ...]
    historical_worst_case_win_rate: float
    script_average_win_rate: float
    full_objective_rate: float
    config_hash: str
    pool_manifest_sha256: str
    payoff_report_sha256: str
    scenario_hash: str

    def __post_init__(self) -> None:
        if _POLICY_ID.fullmatch(self.policy_id) is None:
            raise ValueError("Invalid Strong Base policy ID")
        if not _relative_path(self.checkpoint) or not _relative_path(
            self.validation_report
        ):
            raise ValueError("Strong Base evidence paths must be relative")
        if not _validation_only_path(PurePosixPath(self.validation_report)):
            raise ValueError("Strong Base selection requires validation-only paths")
        hashes = (
            self.checkpoint_sha256,
            self.validation_report_sha256,
            self.config_hash,
            self.pool_manifest_sha256,
            self.payoff_report_sha256,
        )
        if any(_SHA256.fullmatch(value) is None for value in hashes):
            raise ValueError("Strong Base selection hashes must be SHA-256 values")
        if self.environment_steps < 0 or not self.scenario_hash:
            raise ValueError("Invalid Strong Base candidate provenance")
        if self.integrity_passed == bool(self.rejection_reasons):
            raise ValueError("Candidate integrity and rejection reasons disagree")
        rates = (
            self.historical_worst_case_win_rate,
            self.script_average_win_rate,
            self.full_objective_rate,
        )
        if any(not _probability(value) for value in rates):
            raise ValueError("Candidate metrics must be finite probabilities")
        if any(not reason for reason in self.rejection_reasons):
            raise ValueError("Candidate rejection reasons must be non-empty")


@dataclass(frozen=True)
class StrongBaseSelection:
    selected: StrongBaseCandidate
    candidates: tuple[dict[str, object], ...]
    selection_rule: tuple[str, ...] = SELECTION_RULE


def _rank(candidate: StrongBaseCandidate) -> tuple[float | int, ...]:
    return (
        int(candidate.integrity_passed),
        candidate.historical_worst_case_win_rate,
        candidate.script_average_win_rate,
        candidate.full_objective_rate,
        -candidate.environment_steps,
    )


def select_strong_base(
    candidates: Sequence[StrongBaseCandidate],
) -> StrongBaseSelection:
    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.policy_id))
    if not ordered:
        raise ValueError("Strong Base selection requires candidates")
    if len({candidate.policy_id for candidate in ordered}) != len(ordered):
        raise ValueError("Strong Base candidate policy IDs must be unique")
    if len({candidate.checkpoint_sha256 for candidate in ordered}) != len(ordered):
        raise ValueError("Strong Base candidate checkpoint hashes must be unique")
    eligible = tuple(candidate for candidate in ordered if candidate.integrity_passed)
    if not eligible:
        raise ValueError("No Strong Base candidate passes integrity constraints")
    best_rank = max(_rank(candidate) for candidate in eligible)
    winners = tuple(candidate for candidate in eligible if _rank(candidate) == best_rank)
    if len(winners) != 1:
        raise ValueError("Strong Base candidates have an unresolved exact tie")
    rows = tuple(
        {
            **asdict(candidate),
            "rejection_reasons": list(candidate.rejection_reasons),
            "eligible": candidate.integrity_passed,
        }
        for candidate in ordered
    )
    return StrongBaseSelection(selected=winners[0], candidates=rows)


def load_candidate_evidence(
    path: Path, *, artifact_root: Path
) -> StrongBaseCandidate:
    root = artifact_root.expanduser().resolve()
    evidence_path = path.expanduser().resolve()
    try:
        relative_report = evidence_path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("Candidate evidence must be inside the artifact root") from error
    if not _validation_only_path(PurePosixPath(relative_report)):
        raise ValueError("Strong Base selection requires validation-only paths")
    payload: Any = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != _EVIDENCE_FIELDS:
        raise ValueError("Strong Base candidate evidence fields do not match schema")
    if (
        payload["schema_version"] != 1
        or payload["split"] != "validation"
        or payload["test_cases_accessed"] is not False
    ):
        raise ValueError("Strong Base selection requires validation-only evidence")
    checkpoint_relative = payload["checkpoint"]
    if not isinstance(checkpoint_relative, str) or not _relative_path(checkpoint_relative):
        raise ValueError("Candidate checkpoint path must be relative")
    checkpoint = root / checkpoint_relative
    if (
        not checkpoint.is_file()
        or sha256_file(checkpoint) != payload["checkpoint_sha256"]
    ):
        raise ValueError("Candidate checkpoint hash does not match")
    return StrongBaseCandidate(
        policy_id=payload["policy_id"],
        checkpoint=checkpoint_relative,
        checkpoint_sha256=payload["checkpoint_sha256"],
        validation_report=relative_report,
        validation_report_sha256=sha256_file(evidence_path),
        environment_steps=payload["environment_steps"],
        integrity_passed=payload["integrity_passed"],
        rejection_reasons=tuple(payload["rejection_reasons"]),
        historical_worst_case_win_rate=payload["historical_worst_case_win_rate"],
        script_average_win_rate=payload["script_average_win_rate"],
        full_objective_rate=payload["full_objective_rate"],
        config_hash=payload["config_hash"],
        pool_manifest_sha256=payload["pool_manifest_sha256"],
        payoff_report_sha256=payload["payoff_report_sha256"],
        scenario_hash=payload["scenario_hash"],
    )
