from __future__ import annotations

import pytest

from botcolosseo.envs.extraction_rules import (
    EXTRACTION_REQUIRED_TICS,
    Backpack,
    CorpseCache,
    LifeState,
    PlayerRaidState,
    advance_extraction,
    winner,
)


def test_backpack_picks_up_and_replaces_oldest_minimum() -> None:
    backpack = Backpack((10, 10, 25))

    result = backpack.pickup(50)

    assert result.accepted is True
    assert result.dropped_value == 10
    assert result.backpack.items == (50, 10, 25)
    assert result.backpack.total == 85
    assert result.backpack.minimum_value == 10
    assert result.backpack.free_slots == 0


def test_backpack_rejects_non_improving_item() -> None:
    backpack = Backpack((10, 25, 50))

    result = backpack.pickup(10)

    assert result.accepted is False
    assert result.backpack is backpack
    assert result.dropped_value is None


def test_five_fixed_hits_create_exact_value_cache_without_respawn() -> None:
    player = PlayerRaidState(backpack=Backpack((10, 25, 50)))

    for expected_health in (80, 60, 40, 20):
        result = player.take_hit()
        assert result.cache is None
        assert result.player.health == expected_health
        player = result.player
    terminal = player.take_hit()

    assert terminal.player.life_state is LifeState.DEAD
    assert terminal.player.health == 0
    assert terminal.player.backpack.items == ()
    assert terminal.cache == CorpseCache((10, 25, 50))
    assert terminal.cache.total == 85
    with pytest.raises(ValueError, match="Inactive player"):
        terminal.player.take_hit()


def test_cache_processes_high_value_first_and_conserves_rejected_value() -> None:
    cache = CorpseCache((10, 25, 50))

    result = cache.loot(Backpack((10, 25, 25)))

    assert result.backpack.items == (50, 25, 25)
    assert result.world_drops == (10,)
    assert result.cache.items == (25, 10)
    assert (
        result.backpack.total
        + sum(result.world_drops)
        + result.cache.total
        == Backpack((10, 25, 25)).total + cache.total
    )


def test_only_extraction_banks_carried_value() -> None:
    active = PlayerRaidState(backpack=Backpack((10, 25, 50)))

    extracted = active.extract()

    assert active.banked_value == 0
    assert extracted.life_state is LifeState.EXTRACTED
    assert extracted.banked_value == 85
    assert extracted.backpack.items == ()
    assert winner(extracted.banked_value, 30) == 1
    assert winner(0, 0) == 3


def test_extraction_requires_continuous_safe_progress() -> None:
    almost = advance_extraction(
        0,
        elapsed_tics=EXTRACTION_REQUIRED_TICS - 1,
        extraction_open=True,
        inside_zone=True,
    )
    completed = advance_extraction(
        almost.tics,
        elapsed_tics=1,
        extraction_open=True,
        inside_zone=True,
    )
    interrupted = advance_extraction(
        almost.tics,
        elapsed_tics=1,
        extraction_open=True,
        inside_zone=True,
        took_damage=True,
    )

    assert almost.completed is False
    assert completed.completed is True
    assert completed.tics == EXTRACTION_REQUIRED_TICS
    assert interrupted.interrupted is True
    assert interrupted.tics == 0
