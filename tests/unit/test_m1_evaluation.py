from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from botcolosseo.evaluation.m1 import (
    M1_TEACHERS,
    M1_THRESHOLDS,
    evaluate_cases,
    wilson_interval,
    write_evidence,
)
from botcolosseo.scenarios.splits import EpisodeCase, TaskKind


@pytest.mark.parametrize(
    ("successes", "expected"),
    ((0, (0.0, 0.03699)), (75, (0.65696, 0.82455)), (100, (0.96301, 1.0))),
)
def test_wilson_interval_matches_known_values(
    successes: int, expected: tuple[float, float]
) -> None:
    assert wilson_interval(successes, 100) == pytest.approx(expected, abs=1e-5)


@dataclass(frozen=True)
class FakeEpisode:
    success: bool
    truncated: bool
    decisions: int
    total_reward: float
    event_types: tuple[str, ...]
    scenario_hash: str = "sha256:test"


def _cases() -> tuple[EpisodeCase, ...]:
    return tuple(
        EpisodeCase("test", task, 100 + index, 0, 0, "direct_upper")
        for index, task in enumerate(TaskKind)
    )


def test_evaluator_runs_every_case_once_and_maps_teachers() -> None:
    calls: list[tuple[int, str]] = []

    def runner(case: EpisodeCase, teacher: str) -> FakeEpisode:
        calls.append((case.seed, teacher))
        return FakeEpisode(True, False, 4, 1.0, ("task_success",))

    summary, rows = evaluate_cases(_cases(), runner=runner, official=False)

    assert calls == [(case.seed, M1_TEACHERS[case.task]) for case in _cases()]
    assert [row["seed"] for row in rows] == [case.seed for case in _cases()]
    assert summary.official is False
    assert summary.passed is False
    assert summary.protocol_inconsistencies == 0


def test_gate_fails_threshold_miss_or_protocol_inconsistency() -> None:
    cases = tuple(
        EpisodeCase("test", TaskKind.NAVIGATION, seed, 0, 0, "direct_upper")
        for seed in range(100)
    )

    def runner(case: EpisodeCase, teacher: str) -> FakeEpisode:
        del teacher
        if case.seed == 0:
            return FakeEpisode(True, False, 4, 1.0, ())
        success = case.seed < 94
        events = ("task_success",) if success else ()
        return FakeEpisode(success, not success, 4, 1.0, events)

    summary, _ = evaluate_cases(cases, runner=runner, official=True)

    result = summary.capabilities[TaskKind.NAVIGATION.value]
    assert result.successes == 94
    assert result.threshold == M1_THRESHOLDS[TaskKind.NAVIGATION]
    assert result.passed is False
    assert summary.protocol_inconsistencies == 1
    assert summary.passed is False


def test_evidence_csv_uses_repository_safe_line_endings(tmp_path: Path) -> None:
    summary, rows = evaluate_cases(
        _cases(),
        runner=lambda case, teacher: FakeEpisode(True, False, 1, 1.0, ("task_success",)),
        official=False,
    )

    write_evidence(tmp_path, summary, rows, {"official": False})

    assert b"\r\n" not in (tmp_path / "episodes.csv").read_bytes()
