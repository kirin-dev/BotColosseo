from collections import Counter

import pytest

from botcolosseo.agents.league_opponents import OpponentSpec
from botcolosseo.scenarios.league_splits import generate_league_splits
from botcolosseo.training.historical_pool import HistoricalPoolManifest, PoolEntry
from botcolosseo.training.league_schedule import LeagueSchedule


def _entry(index: int, *, anchor: bool = False) -> PoolEntry:
    return PoolEntry(
        policy_id=f"policy-{index:02d}",
        checkpoint=f"runs/policy-{index:02d}.pt",
        checkpoint_sha256=f"{index + 1:064x}",
        scenario_hash="scenario",
        config_hash="config",
        source_git_commit="a" * 40,
        parent_checkpoint_sha256="b" * 64,
        environment_steps=index * 200_000,
        admitted_at_utc=f"2026-07-21T{index:02d}:00:00Z",
        validation_report=f"reports/validation-{index:02d}.json",
        validation_report_sha256=f"{index + 101:064x}",
        script_average_win_rate=0.75,
        script_worst_case_win_rate=0.60,
        objective_rate=0.90,
        payoff_by_policy={"axis": index / 10},
        anchor=anchor,
        admission_reason="anchor" if anchor else "diversity",
    )


def _pool(count: int) -> HistoricalPoolManifest:
    return HistoricalPoolManifest(
        schema_version=1,
        pool_version=0,
        parent_manifest_sha256=None,
        created_at_utc="2026-07-21T00:00:00Z",
        entries=tuple(_entry(index, anchor=index == 0) for index in range(count)),
    )


def _scripts() -> tuple[OpponentSpec, ...]:
    return tuple(
        OpponentSpec(
            opponent_id=name,
            kind="script",
            checkpoint=None,
            checkpoint_sha256=None,
            scenario_hash="scenario",
            selection_evidence=f"builtin:{name}",
        )
        for name in ("objective_first", "aggressive_script", "defensive_script")
    )


def _schedule(pool_count: int = 3) -> LeagueSchedule:
    pool = _pool(pool_count)
    return LeagueSchedule(
        cases=generate_league_splits()["train"],
        scripts=_scripts(),
        pool=pool,
        win_rates={entry.policy_id: index / 4 for index, entry in enumerate(pool.entries)},
        payoff_hash="c" * 64,
        master_seed=20260721,
    )


def test_pair_assignments_share_case_opponent_source_and_probability() -> None:
    host, opponent = _schedule().assignments(37)

    assert host.pair_slot == opponent.pair_slot == 37
    assert (host.case.learner_side, opponent.case.learner_side) == ("host", "opponent")
    assert host.case.seed == opponent.case.seed
    assert host.opponent == opponent.opponent
    assert host.source == opponent.source
    assert host.sampling_probability == opponent.sampling_probability


def test_schedule_exposes_immutable_opponent_descriptor_lookup() -> None:
    schedule = _schedule()

    specs = schedule.opponent_specs

    assert set(specs) == {
        "objective_first",
        "aggressive_script",
        "defensive_script",
        "policy-00",
        "policy-01",
        "policy-02",
    }
    assert specs["policy-00"].kind == "checkpoint"


def test_schedule_is_restart_safe() -> None:
    first = _schedule()
    second = _schedule()

    assert [first.assignments(index) for index in range(1_000)] == [
        second.assignments(index) for index in range(1_000)
    ]


def test_schedule_realizes_frozen_source_mixture() -> None:
    schedule = _schedule()
    counts = Counter(schedule.assignments(index)[0].source for index in range(10_000))

    assert counts["script"] / 10_000 == pytest.approx(0.40, abs=0.02)
    assert counts["pfsp"] / 10_000 == pytest.approx(0.50, abs=0.02)
    assert counts["uniform_history"] / 10_000 == pytest.approx(0.10, abs=0.02)


def test_history_mass_falls_back_to_scripts_before_two_entries() -> None:
    schedule = _schedule(pool_count=1)

    assignments = [schedule.assignments(index)[0] for index in range(100)]

    assert {assignment.source for assignment in assignments} == {"script"}
    assert {assignment.sampling_probability for assignment in assignments} == {1.0}


def test_pfsp_requires_complete_current_payoffs() -> None:
    pool = _pool(2)
    with pytest.raises(ValueError, match="payoff"):
        LeagueSchedule(
            cases=generate_league_splits()["train"],
            scripts=_scripts(),
            pool=pool,
            win_rates={pool.entries[0].policy_id: 0.5},
            payoff_hash="c" * 64,
        )


def test_pfsp_requires_a_valid_payoff_report_hash() -> None:
    pool = _pool(2)
    with pytest.raises(ValueError, match="payoff hash"):
        LeagueSchedule(
            cases=generate_league_splits()["train"],
            scripts=_scripts(),
            pool=pool,
            win_rates={entry.policy_id: 0.5 for entry in pool.entries},
            payoff_hash="not-a-sha256",
        )
