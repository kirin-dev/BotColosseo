from __future__ import annotations

import pytest

from botcolosseo.envs.ipc import WorkerTimeout
from botcolosseo.evaluation.extraction import (
    ExtractionEpisodeMetrics,
    evaluate_extraction_episode_with_retries,
    is_aggressive_showcase_chain,
    summarize_extraction_episodes,
)


def test_extraction_episode_summary_exposes_capability_and_style_metrics() -> None:
    episodes = (
        ExtractionEpisodeMetrics(
            seed=1,
            learner_side="host",
            opponent_style="strong",
            decisions=100,
            extracted_value=85,
            extracted=True,
            died=False,
            valid_hits=5,
            kills=1,
            cache_looted=1,
            attack_decisions=8,
            unique_route_cells=7,
            terminated=True,
            truncated=False,
            won=True,
            opponent_extracted=False,
            opponent_extracted_value=0,
            extracted_value_advantage=85,
        ),
        ExtractionEpisodeMetrics(
            seed=2,
            learner_side="opponent",
            opponent_style="defensive",
            decisions=200,
            extracted_value=0,
            extracted=False,
            died=True,
            valid_hits=1,
            kills=0,
            cache_looted=0,
            attack_decisions=2,
            unique_route_cells=3,
            terminated=True,
            truncated=False,
            won=False,
            opponent_extracted=True,
            opponent_extracted_value=25,
            extracted_value_advantage=-25,
        ),
    )

    summary = summarize_extraction_episodes(episodes)

    assert summary["extraction_rate"] == 0.5
    assert summary["mean_extracted_value"] == 42.5
    assert summary["death_rate"] == 0.5
    assert summary["attack_decisions_mean"] == 5
    assert summary["route_cells_mean"] == 5
    assert summary["valid_hits_total"] == 6
    assert summary["win_rate"] == 0.5
    assert summary["prevent_opponent_extraction_rate"] == 0.5
    assert summary["mean_extracted_value_advantage"] == 30


def test_extraction_episode_summary_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="requires episodes"):
        summarize_extraction_episodes(())


def test_extraction_evaluation_retries_transient_startup_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = ExtractionEpisodeMetrics(
        seed=1,
        learner_side="host",
        opponent_style="strong",
        decisions=1,
        extracted_value=0,
        extracted=False,
        died=False,
        valid_hits=0,
        kills=0,
        cache_looted=0,
        attack_decisions=0,
        unique_route_cells=1,
        terminated=False,
        truncated=True,
    )
    calls = 0

    def fake_evaluate(**kwargs: object) -> ExtractionEpisodeMetrics:
        nonlocal calls
        del kwargs
        calls += 1
        if calls < 3:
            raise WorkerTimeout("port collision")
        return expected

    monkeypatch.setattr(
        "botcolosseo.evaluation.extraction.evaluate_extraction_episode",
        fake_evaluate,
    )

    result = evaluate_extraction_episode_with_retries(startup_attempts=3)

    assert result.environment_attempts == 3
    assert calls == 3


def test_extraction_evaluation_retry_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(**kwargs: object) -> ExtractionEpisodeMetrics:
        del kwargs
        raise WorkerTimeout("port collision")

    monkeypatch.setattr(
        "botcolosseo.evaluation.extraction.evaluate_extraction_episode",
        fail,
    )

    with pytest.raises(WorkerTimeout):
        evaluate_extraction_episode_with_retries(startup_attempts=2)


def test_aggressive_showcase_chain_requires_complete_causal_story() -> None:
    complete = ExtractionEpisodeMetrics(
        seed=1,
        learner_side="host",
        opponent_style="aggressive",
        decisions=300,
        extracted_value=50,
        extracted=True,
        died=False,
        valid_hits=5,
        kills=1,
        cache_looted=1,
        attack_decisions=8,
        unique_route_cells=7,
        terminated=False,
        truncated=False,
    )

    assert is_aggressive_showcase_chain(complete)
    assert not is_aggressive_showcase_chain(
        ExtractionEpisodeMetrics(**{**vars(complete), "extracted": False})
    )
    assert not is_aggressive_showcase_chain(
        ExtractionEpisodeMetrics(**{**vars(complete), "cache_looted": 0})
    )
