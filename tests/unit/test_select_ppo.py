from __future__ import annotations

import pytest

from botcolosseo.cli.select_ppo import CandidateValidation, select_candidate


def _candidate(
    steps: int,
    *,
    objective_rate: float,
    win_rate: float,
    score_difference: float,
    protocol_inconsistencies: int = 0,
) -> CandidateValidation:
    return CandidateValidation(
        checkpoint=f"candidate-{steps:07d}.pt",
        checkpoint_sha256=f"sha-{steps}",
        environment_steps=steps,
        episodes=30,
        objective_rate=objective_rate,
        win_rate=win_rate,
        mean_score_difference=score_difference,
        protocol_inconsistencies=protocol_inconsistencies,
    )


def test_selection_is_validation_lexicographic_and_prefers_earlier_tie() -> None:
    candidates = (
        _candidate(100_000, objective_rate=0.8, win_rate=0.6, score_difference=1.0),
        _candidate(200_000, objective_rate=0.8, win_rate=0.7, score_difference=0.5),
        _candidate(300_000, objective_rate=0.9, win_rate=0.5, score_difference=0.0),
        _candidate(400_000, objective_rate=0.9, win_rate=0.5, score_difference=0.0),
    )

    selected = select_candidate(candidates, expected_episodes=30)

    assert selected.environment_steps == 300_000


def test_selection_rejects_incomplete_or_protocol_dirty_candidate() -> None:
    clean = _candidate(100_000, objective_rate=0.8, win_rate=0.6, score_difference=1.0)
    with pytest.raises(ValueError, match="episode count"):
        select_candidate((clean,), expected_episodes=50)
    with pytest.raises(ValueError, match="protocol"):
        select_candidate(
            (
                _candidate(
                    100_000,
                    objective_rate=0.8,
                    win_rate=0.6,
                    score_difference=1.0,
                    protocol_inconsistencies=1,
                ),
            ),
            expected_episodes=30,
        )
