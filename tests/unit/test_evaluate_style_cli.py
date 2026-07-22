from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from botcolosseo.cli.evaluate_style import (
    _run_with_retries,
    build_parser,
    select_style_cases,
)
from botcolosseo.evaluation.m2 import load_duel_cases
from botcolosseo.evaluation.style import StyleEpisodeRecord
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS


def _record() -> StyleEpisodeRecord:
    return StyleEpisodeRecord(
        policy="aggressive",
        split="validation",
        opponent="fixed_route",
        pair_index=1,
        seed=2,
        learner_side="host",
        outcome="draw",
        objective_completed=False,
        learner_score=0,
        opponent_score=0,
        decisions=10,
        attack_decisions=0,
        valid_hits=0,
        engagement_initiations=0,
        forward_hits=0,
        invalid_attack_decisions=0,
        objective_chase_decisions=0,
        retreat_decisions=0,
        terminated=False,
        truncated=True,
        peer_tic_lag_max=0,
        protocol_inconsistent=False,
        action_tic_inconsistent=False,
        score_event_inconsistent=False,
        scenario_hash="scenario",
    )


def test_style_case_selection_is_balanced_side_swapped_and_validation_only() -> None:
    cases = load_duel_cases(
        Path("configs/m2/validation.json"),
        expected_split="validation",
        pairs_per_opponent=50,
    )

    selected = select_style_cases(cases, pairs_per_opponent=2)

    assert len(selected) == 20
    assert {case.split for case in selected} == {"validation"}
    assert {
        opponent: sum(case.opponent == opponent for case in selected)
        for opponent in DUEL_OPPONENTS
    } == {opponent: 4 for opponent in DUEL_OPPONENTS}
    for host, opponent in zip(selected[::2], selected[1::2], strict=True):
        assert (host.pair_index, host.seed) == (opponent.pair_index, opponent.seed)
        assert (host.learner_side, opponent.learner_side) == ("host", "opponent")


def test_style_retry_is_bounded_and_records_attempts() -> None:
    attempts = 0

    def runner() -> StyleEpisodeRecord:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("Duel respawn did not complete within the warm-up limit")
        return _record()

    result = _run_with_retries(runner, max_attempts=2)

    assert result == replace(_record(), environment_attempts=2)
    with pytest.raises(RuntimeError, match="unrelated"):
        _run_with_retries(
            lambda: (_ for _ in ()).throw(RuntimeError("unrelated")),
            max_attempts=2,
        )


def test_style_cli_requires_explicit_checkpoints_and_output() -> None:
    args = build_parser().parse_args(
        [
            "--base-checkpoint",
            "base.pt",
            "--aggressive-checkpoint",
            "aggressive.pt",
            "--output-dir",
            "reports/m4/validation",
        ]
    )

    assert args.base_checkpoint == Path("base.pt")
    assert args.aggressive_checkpoint == Path("aggressive.pt")
    assert args.pairs_per_opponent == 10
