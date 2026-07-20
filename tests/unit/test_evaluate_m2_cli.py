from __future__ import annotations

from pathlib import Path

import pytest

from botcolosseo.cli.evaluate_m2 import (
    ensure_evidence_targets_absent,
    run_case_with_retries,
)
from botcolosseo.evaluation.m2 import M2EpisodeRecord


def _record() -> M2EpisodeRecord:
    return M2EpisodeRecord(
        policy="ppo",
        split="validation",
        opponent="random_legal",
        pair_index=1,
        seed=2,
        learner_side="host",
        outcome="draw",
        objective_completed=False,
        learner_score=0,
        opponent_score=0,
        decisions=4,
        terminated=False,
        truncated=True,
        peer_tic_lag_max=0,
        protocol_inconsistent=False,
        scenario_hash="scenario",
    )


def test_output_directory_allows_unrelated_m2_reports(tmp_path: Path) -> None:
    (tmp_path / "ppo-training-summary.json").write_text("{}\n")

    ensure_evidence_targets_absent(tmp_path)

    (tmp_path / "summary.json").write_text("{}\n")
    with pytest.raises(FileExistsError, match="summary.json"):
        ensure_evidence_targets_absent(tmp_path)


def test_transient_respawn_failure_retries_same_case_once() -> None:
    attempts = 0

    def runner() -> M2EpisodeRecord:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("Duel respawn did not complete within the warm-up limit")
        return _record()

    result = run_case_with_retries(runner, max_attempts=2)

    assert attempts == 2
    assert result.environment_attempts == 2


def test_nontransient_error_is_not_retried() -> None:
    attempts = 0

    def runner() -> M2EpisodeRecord:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("scenario hash mismatch")

    with pytest.raises(RuntimeError, match="scenario hash mismatch"):
        run_case_with_retries(runner, max_attempts=2)
    assert attempts == 1
