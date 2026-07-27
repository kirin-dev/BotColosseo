from __future__ import annotations

from botcolosseo.evaluation.extraction import ExtractionEpisodeMetrics
from botcolosseo.evaluation.extraction_gates import (
    strong_validation_gate,
    style_validation_gate,
)


def episodes(
    count: int,
    *,
    style_delta: int = 0,
) -> tuple[ExtractionEpisodeMetrics, ...]:
    opponent_styles = ("strong", "aggressive", "defensive", "explorer")
    return tuple(
        ExtractionEpisodeMetrics(
            seed=index // 2,
            learner_side="host" if index % 2 == 0 else "opponent",
            opponent_style=opponent_styles[index % 4],
            decisions=100,
            extracted_value=50,
            extracted=True,
            died=False,
            valid_hits=2 + style_delta,
            kills=0,
            cache_looted=1,
            attack_decisions=5,
            unique_route_cells=5,
            terminated=True,
            truncated=False,
            won=True,
            opponent_extracted=False,
            extracted_value_advantage=50,
        )
        for index in range(count)
    )


def test_strong_gate_passes_complete_high_capability_evidence() -> None:
    solo = tuple(
        ExtractionEpisodeMetrics(
            **{
                **episode.__dict__,
                "opponent_style": "idle",
            }
        )
        for episode in episodes(40)
    )
    result = strong_validation_gate(episodes(240), episodes(120), solo)
    assert result.passed


def test_aggressive_style_gate_requires_paired_positive_ci() -> None:
    result = style_validation_gate(
        style="aggressive",
        strong=episodes(240),
        styled=episodes(240, style_delta=1),
    )
    assert result.passed
    assert all(check.passed for check in result.checks)
