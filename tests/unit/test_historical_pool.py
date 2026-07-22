from dataclasses import replace
from pathlib import Path

import pytest

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.training.historical_pool import (
    AdmissionMetrics,
    HistoricalPoolManifest,
    PoolEntry,
    admission_decision,
    admit_candidate,
    load_pool,
    write_pool_atomic,
)


def _artifact(root: Path, relative: str, content: str) -> tuple[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return relative, sha256_file(path)


def _entry(
    root: Path,
    index: int,
    *,
    anchor: bool = False,
    payoffs: dict[str, float] | None = None,
) -> PoolEntry:
    checkpoint, checkpoint_hash = _artifact(
        root, f"models/policy-{index:02d}.pt", f"checkpoint-{index}"
    )
    report, report_hash = _artifact(
        root, f"reports/validation-{index:02d}.json", f"report-{index}"
    )
    return PoolEntry(
        policy_id=f"policy-{index:02d}",
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        scenario_hash="scenario",
        config_hash="config",
        source_git_commit="a" * 40,
        parent_checkpoint_sha256="b" * 64,
        environment_steps=index * 200_000,
        admitted_at_utc=f"2026-07-21T{index:02d}:00:00Z",
        validation_report=report,
        validation_report_sha256=report_hash,
        script_average_win_rate=0.75,
        script_worst_case_win_rate=0.60,
        objective_rate=0.90,
        payoff_by_policy=payoffs or {"axis-a": index / 20, "axis-b": 0.5},
        anchor=anchor,
        admission_reason="m2_anchor" if anchor else "diversity",
    )


def _pool(root: Path, count: int = 1) -> HistoricalPoolManifest:
    return HistoricalPoolManifest(
        schema_version=1,
        pool_version=0,
        parent_manifest_sha256=None,
        created_at_utc="2026-07-21T00:00:00Z",
        entries=tuple(_entry(root, index, anchor=index == 0) for index in range(count)),
    )


def _metrics(pool: HistoricalPoolManifest) -> AdmissionMetrics:
    active = {
        entry.policy_id: {"axis-a": index / 20, "axis-b": 0.5}
        for index, entry in enumerate(pool.entries)
    }
    return AdmissionMetrics(
        integrity_ok=True,
        validation_complete=True,
        paired_side_swapped=True,
        protocol_inconsistencies=0,
        source_split="validation",
        candidate_script_average=0.70,
        active_script_average=0.75,
        candidate_historical_worst_case=0.55,
        active_historical_worst_case=0.50,
        candidate_payoffs={"axis-a": 0.95, "axis-b": 0.95},
        active_payoffs=active,
    )


def test_pool_manifest_round_trip_hash_and_artifact_verification(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    path = tmp_path / "reports/m3/pool.json"

    write_pool_atomic(pool, path)
    loaded = load_pool(path, artifact_root=tmp_path)

    assert loaded == pool
    assert loaded.manifest_sha256 == pool.manifest_sha256
    assert not list(path.parent.glob("*.tmp"))

    checkpoint = tmp_path / pool.entries[0].checkpoint
    checkpoint.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint hash"):
        load_pool(path, artifact_root=tmp_path)


def test_pool_rejects_duplicate_ids_and_hashes(tmp_path: Path) -> None:
    anchor = _entry(tmp_path, 0, anchor=True)
    duplicate_id = replace(_entry(tmp_path, 1), policy_id=anchor.policy_id)
    with pytest.raises(ValueError, match="policy IDs"):
        HistoricalPoolManifest(1, 0, None, "2026-07-21T00:00:00Z", (anchor, duplicate_id))

    duplicate_hash = replace(_entry(tmp_path, 2), checkpoint_sha256=anchor.checkpoint_sha256)
    with pytest.raises(ValueError, match="checkpoint hashes"):
        HistoricalPoolManifest(1, 0, None, "2026-07-21T00:00:00Z", (anchor, duplicate_hash))


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"integrity_ok": False}, "integrity"),
        ({"validation_complete": False}, "complete"),
        ({"paired_side_swapped": False}, "paired"),
        ({"protocol_inconsistencies": 1}, "protocol"),
        ({"source_split": "test"}, "validation"),
        ({"candidate_script_average": 0.64}, "script"),
        (
            {
                "candidate_historical_worst_case": 0.40,
                "candidate_payoffs": {"axis-a": 0.01, "axis-b": 0.5},
            },
            "diversity",
        ),
    ],
)
def test_each_admission_constraint_fails_closed(
    tmp_path: Path, change: dict[str, object], reason: str
) -> None:
    pool = _pool(tmp_path, 2)
    metrics = replace(_metrics(pool), **change)
    candidate = _entry(tmp_path, 12, payoffs=metrics.candidate_payoffs)

    decision = admission_decision(pool, candidate, metrics)

    assert decision.eligible is False
    assert reason in decision.reason


def test_admission_returns_new_pool_without_mutating_old_version(tmp_path: Path) -> None:
    pool = _pool(tmp_path, 2)
    metrics = _metrics(pool)
    candidate = _entry(tmp_path, 12, payoffs=metrics.candidate_payoffs)

    updated = admit_candidate(pool, candidate, metrics)

    assert len(pool.entries) == 2
    assert len(updated.entries) == 3
    assert updated.entries[-1] == candidate
    assert updated.pool_version == pool.pool_version + 1
    assert updated.parent_manifest_sha256 == pool.manifest_sha256


def test_full_pool_replacement_is_deterministic_and_protects_anchor_and_newest(
    tmp_path: Path,
) -> None:
    pool = _pool(tmp_path, 12)
    metrics = _metrics(pool)
    candidate = _entry(tmp_path, 12, payoffs=metrics.candidate_payoffs)

    first = admit_candidate(pool, candidate, metrics)
    second = admit_candidate(pool, candidate, metrics)

    assert first == second
    assert len(first.entries) == 12
    assert pool.entries[0] in first.entries
    assert pool.entries[-1] in first.entries
    assert candidate in first.entries


def test_admission_rejects_payoff_vector_not_bound_to_entry(tmp_path: Path) -> None:
    pool = _pool(tmp_path, 2)
    metrics = _metrics(pool)
    candidate = _entry(tmp_path, 12, payoffs={"axis-a": 0.20, "axis-b": 0.20})

    decision = admission_decision(pool, candidate, metrics)

    assert decision.eligible is False
    assert "payoff evidence" in decision.reason
