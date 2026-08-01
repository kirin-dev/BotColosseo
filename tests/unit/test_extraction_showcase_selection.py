from __future__ import annotations

from botcolosseo.cli.select_extraction_showcases import (
    _evidence_tier,
    _representative,
    _score,
)
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
        aggressive_chains=1,
        extracted_value=30,
    )
    assert _score("aggressive", chain) > _score("aggressive", generic)


def test_showcase_selection_uses_distinct_visible_style_signatures() -> None:
    assert _score(
        "defensive",
        episode(
            successful_disengagements=1,
            meaningful_extractions=1,
        ),
    ) > _score(
        "defensive",
        episode(attack_decisions=0),
    )
    assert _score(
        "explorer",
        episode(
            meaningful_loot_regions=3,
            backpack_upgrades=1,
            upgrade_to_extraction_conversions=1,
        ),
    ) > _score(
        "explorer",
        episode(unique_route_cells=20),
    )


def test_explorer_prefers_no_combat_upgrade_story_over_noisy_combat() -> None:
    strong = episode(
        meaningful_loot_regions=1,
        backpack_upgrades=0,
        upgrade_to_extraction_conversions=0,
    )
    quiet = episode(
        decisions=300,
        meaningful_loot_regions=4,
        backpack_upgrades=1,
        upgrade_to_extraction_conversions=1,
        valid_hits=0,
        kills=0,
    )
    noisy = episode(
        decisions=300,
        meaningful_loot_regions=4,
        backpack_upgrades=1,
        upgrade_to_extraction_conversions=1,
        valid_hits=4,
        kills=1,
    )

    assert _representative("explorer", quiet, strong)
    assert _score("explorer", quiet, strong) > _score(
        "explorer", noisy, strong
    )


def test_defensive_requires_disengagement_to_meaningful_extraction() -> None:
    strong = episode(successful_disengagements=0, meaningful_extractions=1)
    merely_passive = episode(
        decisions=300,
        successful_disengagements=0,
        meaningful_extractions=1,
        attack_decisions=0,
    )
    complete = episode(
        decisions=300,
        successful_disengagements=1,
        disengagement_opportunities=1,
        meaningful_extractions=1,
        attack_decisions=0,
    )

    assert not _representative("defensive", merely_passive, strong)
    assert _representative("defensive", complete, strong)


def test_showcase_selection_accepts_product_only_strong_manifest() -> None:
    checkpoint_sha256 = "a" * 64
    report = {"checkpoint_sha256": checkpoint_sha256}
    manifest = {
        "policy": "strong",
        "admission_kind": "strong_product_showcase",
        "product_showcase_eligible": True,
        "research_gate_passed": False,
        "official_test_eligible": False,
        "showcase_checkpoint_sha256": checkpoint_sha256,
        "test_cases_accessed": False,
    }

    assert _evidence_tier("strong", report, manifest) == "product_showcase"
