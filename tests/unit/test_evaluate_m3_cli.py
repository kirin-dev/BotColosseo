from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from botcolosseo.cli.evaluate_m3 import (
    EvaluationJob,
    append_episode_row,
    build_official_jobs,
    load_episode_rows,
    run_case_with_retries,
    run_resumable_rows,
)
from botcolosseo.evaluation.m3 import M3EpisodeRecord, expected_m3_episode_count
from botcolosseo.scenarios.league_splits import generate_league_splits


def _row(job: EvaluationJob) -> M3EpisodeRecord:
    return M3EpisodeRecord(
        policy=job.policy,
        category=job.category,
        split=job.case.split,
        opponent=job.opponent,
        pair_index=job.case.pair_index,
        seed=job.case.seed,
        learner_side=job.case.learner_side,
        outcome="win",
        objective_completed=True,
        goal_reached=True,
        pickup_completed=True,
        return_completed=True,
        valid_hit=True,
        disengage_success=True,
        learner_score=1,
        opponent_score=0,
        actual_core_x=0.0,
        actual_core_y=0.0,
        decisions=10,
        terminated=True,
        truncated=False,
        peer_tic_lag_max=0,
        protocol_inconsistent=False,
        action_tic_inconsistent=False,
        score_event_inconsistent=False,
        fairness_schema_inconsistent=False,
        scenario_hash="scenario",
        environment_attempts=1,
    )


def _jobs(count: int = 20) -> tuple[EvaluationJob, ...]:
    cases = generate_league_splits()["test"][:count]
    return tuple(
        EvaluationJob(index, "strong_base", "script", "objective_first", case)
        for index, case in enumerate(cases)
    )


def test_official_schedule_has_frozen_counts_and_split_isolation() -> None:
    splits = generate_league_splits()
    historical = tuple(f"historical-{index:02d}" for index in range(8))

    jobs = build_official_jobs(
        splits["test"], splits["heldout"], historical_policy_ids=historical
    )

    assert len(jobs) == expected_m3_episode_count(8)
    assert sum(job.category == "script" for job in jobs) == 500
    assert sum(job.category == "no_opponent" for job in jobs) == 100
    assert sum(job.category == "heldout" for job in jobs) == 100
    assert sum(job.category == "historical" for job in jobs) == 640
    assert all(
        job.case.split == ("heldout" if job.category == "heldout" else "test")
        for job in jobs
    )
    assert {job.case.split for job in jobs} == {"test", "heldout"}


def test_interrupted_resume_is_byte_identical_to_uninterrupted_rows(
    tmp_path: Path,
) -> None:
    jobs = _jobs()
    identity = {"schema_version": 1, "run": "same"}
    resumed = tmp_path / "resumed/episodes.jsonl"
    uninterrupted = tmp_path / "uninterrupted/episodes.jsonl"

    partial, complete = run_resumable_rows(
        jobs,
        runner=_row,
        ledger_path=resumed,
        run_identity=identity,
        resume=False,
        max_attempts=2,
        stop_after=7,
    )
    assert len(partial) == 7
    assert complete is False
    final, complete = run_resumable_rows(
        jobs,
        runner=_row,
        ledger_path=resumed,
        run_identity=identity,
        resume=True,
        max_attempts=2,
    )
    direct, direct_complete = run_resumable_rows(
        jobs,
        runner=_row,
        ledger_path=uninterrupted,
        run_identity=identity,
        resume=False,
        max_attempts=2,
    )

    assert complete is direct_complete is True
    assert final == direct
    assert resumed.read_bytes() == uninterrupted.read_bytes()
    assert (resumed.parent / "run-identity.json").read_bytes() == (
        uninterrupted.parent / "run-identity.json"
    ).read_bytes()


def test_resume_validates_identity_and_ledger_duplicates_fail_closed(
    tmp_path: Path,
) -> None:
    jobs = _jobs(2)
    ledger = tmp_path / "episodes.jsonl"
    run_resumable_rows(
        jobs,
        runner=_row,
        ledger_path=ledger,
        run_identity={"run": "original"},
        resume=False,
        max_attempts=2,
        stop_after=1,
    )
    with pytest.raises(ValueError, match="identity"):
        run_resumable_rows(
            jobs,
            runner=_row,
            ledger_path=ledger,
            run_identity={"run": "drifted"},
            resume=True,
            max_attempts=2,
        )

    first = load_episode_rows(ledger)[0]
    append_episode_row(ledger, first)
    assert load_episode_rows(ledger) == (first,)
    append_episode_row(ledger, replace(first, learner_score=2))
    with pytest.raises(ValueError, match="conflicting duplicate"):
        load_episode_rows(ledger)


def test_retry_is_bounded_and_preserves_job_identity() -> None:
    job = _jobs(1)[0]
    calls: list[EvaluationJob] = []

    def runner(selected: EvaluationJob) -> M3EpisodeRecord:
        calls.append(selected)
        if len(calls) == 1:
            raise RuntimeError("Duel respawn did not complete within the warm-up limit")
        return _row(selected)

    result = run_case_with_retries(job, runner=runner, max_attempts=2)

    assert calls == [job, job]
    assert result.environment_attempts == 2


def test_preflight_parser_requires_bound_official_artifacts() -> None:
    from botcolosseo.cli.evaluate_m3 import build_parser

    parser = build_parser()
    parsed = parser.parse_args(
        [
            "--preflight",
            "--output-dir",
            "reports/m3/official",
            "--selected-checkpoint",
            "models/strong-base/strong-base.pt",
            "--pool",
            "reports/m3/pool.json",
            "--m2-baseline",
            "runs/m2/ppo-full/selected.pt",
        ]
    )
    assert parsed.preflight is True
    assert parsed.resume is False


def test_preflight_mode_reports_identity_without_starting_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from botcolosseo.cli.evaluate_m3 import main

    monkeypatch.setattr(
        "botcolosseo.cli.evaluate_m3.prepare_official_evaluation",
        lambda args: SimpleNamespace(
            run_identity={"git_dirty": False, "expected_episodes": 1_340}
        ),
    )
    monkeypatch.setattr(
        "botcolosseo.cli.evaluate_m3.M3Runtime",
        lambda *args, **kwargs: pytest.fail("preflight must not start the runtime"),
    )

    result = main(
        [
            "--preflight",
            "--output-dir",
            str(tmp_path / "official"),
            "--selected-checkpoint",
            "strong.pt",
            "--pool",
            "pool.json",
            "--m2-baseline",
            "m2.pt",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert '"preflight_passed": true' in output
    assert not (tmp_path / "official").exists()
