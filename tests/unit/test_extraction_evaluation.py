from __future__ import annotations

import pytest

from botcolosseo.evaluation.extraction import (
    ExtractionEpisodeMetrics,
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
        ),
    )

    summary = summarize_extraction_episodes(episodes)

    assert summary["extraction_rate"] == 0.5
    assert summary["mean_extracted_value"] == 42.5
    assert summary["death_rate"] == 0.5
    assert summary["attack_decisions_mean"] == 5
    assert summary["route_cells_mean"] == 5
    assert summary["valid_hits_total"] == 6


def test_extraction_episode_summary_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="requires episodes"):
        summarize_extraction_episodes(())
