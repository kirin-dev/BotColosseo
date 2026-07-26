from __future__ import annotations

from botcolosseo.cli.select_extraction_showcases import _score
from botcolosseo.evaluation.extraction import ExtractionEpisodeMetrics


def episode(**overrides) -> ExtractionEpisodeMetrics:
    values = {
        "seed": 1,
        "learner_side": "host",
        "opponent_style": "strong",
        "decisions": 150,
        "extracted_value": 50,
        "extracted": True,
        "died": False,
        "valid_hits": 0,
        "kills": 0,
        "cache_looted": 0,
        "attack_decisions": 0,
        "unique_route_cells": 5,
        "terminated": True,
        "truncated": False,
        "loot_pickups": 3,
        "won": True,
    }
    values.update(overrides)
    return ExtractionEpisodeMetrics(**values)


def test_showcase_selection_prefers_complete_aggressive_causal_chain() -> None:
    generic = episode(extracted_value=85)
    chain = episode(
        valid_hits=5,
        kills=1,
        cache_looted=1,
        extracted_value=30,
    )
    assert _score("aggressive", chain) > _score("aggressive", generic)


def test_showcase_selection_uses_distinct_visible_style_signatures() -> None:
    assert _score("defensive", episode(attack_decisions=0)) > _score(
        "defensive",
        episode(attack_decisions=10),
    )
    assert _score("explorer", episode(unique_route_cells=12)) > _score(
        "explorer",
        episode(unique_route_cells=4),
    )
