from __future__ import annotations

from dataclasses import replace

import pytest

from botcolosseo.evaluation.extraction import ExtractionEpisodeMetrics
from botcolosseo.evaluation.extraction_gates import (
    aggressive_showcase_direction_counts,
    directional_showcase_heldout_gate,
    strong_validation_gate,
    style_heldout_gate,
    style_showcase_direction_counts,
    style_validation_gate,
)


def episodes(
    count: int,
    *,
    style: str | None = None,
) -> tuple[ExtractionEpisodeMetrics, ...]:
    opponent_styles = ("strong", "aggressive", "defensive", "explorer")
    base = tuple(
        ExtractionEpisodeMetrics(
            seed=index // 2,
            learner_side="host" if index % 2 == 0 else "opponent",
            opponent_style=opponent_styles[(index // 2) % 4],
            decisions=100,
            extracted_value=50,
            extracted=True,
            died=False,
            valid_hits=2,
            kills=0,
            cache_looted=1,
            attack_decisions=5,
            unique_route_cells=5,
            terminated=True,
            truncated=False,
            won=True,
            opponent_extracted=False,
            extracted_value_advantage=50,
            encounter_opportunities=1,
            meaningful_extractions=1,
            meaningful_loot_regions=1,
        )
        for index in range(count)
    )
    if style == "aggressive":
        return tuple(
            replace(
                item,
                valid_hits=3,
                kills=1,
                favorable_encounter_initiations=1,
                kill_to_cache_conversions=1,
                cache_to_extraction_conversions=1,
                aggressive_chains=1,
            )
            for item in base
        )
    if style == "defensive":
        return tuple(
            replace(
                item,
                disengagement_opportunities=1,
                successful_disengagements=1,
            )
            for item in base
        )
    if style == "explorer":
        return tuple(
            replace(
                item,
                meaningful_loot_regions=3,
                backpack_upgrades=1,
                upgrade_to_extraction_conversions=1,
            )
            for item in base
        )
    return base


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
        styled=episodes(240, style="aggressive"),
    )
    assert result.passed
    assert all(check.passed for check in result.checks)


def test_aggressive_showcase_counts_paired_direction_and_chains() -> None:
    strong = episodes(240)
    styled = list(strong)
    styled[0] = replace(
        styled[0],
        valid_hits=3,
        kills=1,
        kill_to_cache_conversions=1,
        cache_to_extraction_conversions=1,
        aggressive_chains=1,
    )
    styled[1] = replace(styled[1], valid_hits=1)

    counts = aggressive_showcase_direction_counts(strong, tuple(styled))

    assert counts == {
        "positive_pairs": 1,
        "negative_pairs": 1,
        "unchanged_pairs": 238,
        "new_complete_chains": 1,
        "lost_complete_chains": 0,
    }


def test_aggressive_showcase_rejects_duplicate_pair_identity() -> None:
    styled = list(episodes(240, style="aggressive"))
    styled[-1] = styled[0]

    try:
        aggressive_showcase_direction_counts(episodes(240), tuple(styled))
    except ValueError as error:
        assert "uniquely paired" in str(error)
    else:
        raise AssertionError("duplicate paired evidence was accepted")


def test_explorer_showcase_counts_direction_and_upgrade_chains() -> None:
    strong = episodes(240)
    styled = list(strong)
    styled[0] = replace(
        styled[0],
        meaningful_loot_regions=3,
        backpack_upgrades=1,
        upgrade_to_extraction_conversions=1,
    )
    styled[1] = replace(styled[1], meaningful_loot_regions=0)

    counts = style_showcase_direction_counts(
        style="explorer",
        strong=strong,
        styled=tuple(styled),
    )

    assert counts == {
        "showcase_chain_kind": "upgrade_to_extraction",
        "positive_pairs": 1,
        "negative_pairs": 1,
        "unchanged_pairs": 238,
        "new_showcase_chains": 1,
        "lost_showcase_chains": 0,
    }


def test_defensive_showcase_requires_disengagement_and_extraction_chain() -> None:
    strong = episodes(240)
    styled = list(strong)
    styled[0] = replace(
        styled[0],
        disengagement_opportunities=1,
        successful_disengagements=1,
    )

    counts = style_showcase_direction_counts(
        style="defensive",
        strong=strong,
        styled=tuple(styled),
    )

    assert counts["positive_pairs"] == 1
    assert counts["new_showcase_chains"] == 1


def test_style_gate_rejects_aggressive_hits_without_conversion() -> None:
    hit_only = tuple(
        replace(item, valid_hits=4, favorable_encounter_initiations=1)
        for item in episodes(240)
    )

    result = style_validation_gate(
        style="aggressive",
        strong=episodes(240),
        styled=hit_only,
    )

    assert not result.passed
    assert not next(
        check
        for check in result.checks
        if check.name == "anti_hack_complete_combat_chain"
    ).passed


def test_style_gate_rejects_defensive_inactivity() -> None:
    inactive = tuple(
        replace(
            item,
            won=False,
            extracted=False,
            extracted_value=0,
            meaningful_extractions=0,
            attack_decisions=0,
        )
        for item in episodes(240)
    )

    result = style_validation_gate(
        style="defensive",
        strong=episodes(240),
        styled=inactive,
    )

    assert not result.passed
    assert not next(
        check for check in result.checks if check.name == "anti_hack_not_inactive"
    ).passed


def test_style_gate_rejects_explorer_wandering_without_upgrade() -> None:
    wandering = tuple(
        replace(
            item,
            unique_route_cells=30,
            meaningful_loot_regions=1,
            backpack_upgrades=0,
            upgrade_to_extraction_conversions=0,
        )
        for item in episodes(240)
    )

    result = style_validation_gate(
        style="explorer",
        strong=episodes(240),
        styled=wandering,
    )

    assert not result.passed
    assert not next(
        check
        for check in result.checks
        if check.name == "anti_hack_real_upgrade_conversion"
    ).passed


def test_style_heldout_gate_rejects_one_opponent_collapse() -> None:
    styled = tuple(
        replace(item, won=False)
        if item.opponent_style == "explorer"
        else item
        for item in episodes(120)
    )

    result = style_heldout_gate(
        strong=episodes(120),
        styled=styled,
    )

    assert not result.passed


def test_directional_showcase_heldout_uses_relative_strong_retention() -> None:
    strong = list(episodes(120))
    styled = list(episodes(120))
    explorer_indices = [
        index
        for index, item in enumerate(strong)
        if item.opponent_style == "explorer"
    ]
    for offset, index in enumerate(explorer_indices):
        strong[index] = replace(strong[index], won=offset < 7)
        styled[index] = replace(styled[index], won=offset < 8)

    result, comparisons = directional_showcase_heldout_gate(
        strong=tuple(strong),
        styled=tuple(styled),
    )

    assert result.passed
    assert comparisons["explorer"]["strong_win_rate"] == 7 / 30
    assert comparisons["explorer"]["styled_win_rate"] == 8 / 30
    assert comparisons["explorer"]["allowed_floor"] == pytest.approx(1 / 30)
    assert comparisons["explorer"]["relative_margin"] == pytest.approx(7 / 30)


def test_directional_showcase_heldout_rejects_real_relative_collapse() -> None:
    styled = tuple(
        replace(item, won=False)
        if item.opponent_style == "defensive"
        else item
        for item in episodes(120)
    )

    result, _ = directional_showcase_heldout_gate(
        strong=episodes(120),
        styled=styled,
    )

    assert not result.passed
