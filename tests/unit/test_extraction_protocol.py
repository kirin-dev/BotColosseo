from __future__ import annotations

from dataclasses import fields, replace

import pytest

from botcolosseo.envs.extraction_protocol import (
    ExtractionEventDecoder,
    ExtractionEventType,
    ExtractionProtocolSnapshot,
)


def snapshot(**changes: int | float) -> ExtractionProtocolSnapshot:
    defaults: dict[str, int | float] = {
        field.name: 0 for field in fields(ExtractionProtocolSnapshot)
    }
    defaults.update(
        {
            "protocol_version": 3,
            "round_state": 1,
            "host_life_state": 1,
            "opponent_life_state": 1,
        }
    )
    defaults.update(changes)
    return ExtractionProtocolSnapshot(**defaults)


def test_protocol_round_trip_preserves_fixed_coordinates_and_public_view() -> None:
    original = snapshot(
        extraction_open=1,
        host_slot_0=10,
        host_slot_1=25,
        host_banked=50,
        host_extraction_zone=2,
        host_extraction_progress=35,
        cache_x=12.5,
        cache_y=-21.25,
    )

    restored = ExtractionProtocolSnapshot.from_values(original.to_values())
    public = restored.public_state("host")

    assert restored == original
    assert public.slots == (10, 25, 0)
    assert public.carried_value == 35
    assert public.free_slots == 1
    assert public.minimum_slot_value == 10
    assert public.extraction_open is True


def test_decoder_emits_counted_value_events() -> None:
    before = snapshot(engine_tic=10)
    after = replace(
        before,
        engine_tic=14,
        host_loot_pickups=1,
        last_loot_value=50,
        last_loot_side=1,
        last_loot_id=7,
        event_serial=1,
    )
    decoder = ExtractionEventDecoder()
    decoder.reset(before)

    events = decoder.decode(after, episode_id=2, decision_index=3)

    assert len(events) == 1
    assert events[0].type is ExtractionEventType.LOOT_PICKUP
    assert events[0].side == "host"
    assert events[0].count == 1
    assert events[0].value == 50


def test_decoder_allows_multiple_cache_items_between_decisions() -> None:
    before = snapshot(engine_tic=10)
    after = replace(
        before,
        engine_tic=14,
        host_cache_looted=3,
        last_loot_value=10,
        event_serial=3,
    )
    decoder = ExtractionEventDecoder()
    decoder.reset(before)

    event = decoder.decode(after, episode_id=0, decision_index=1)[0]

    assert event.type is ExtractionEventType.CACHE_LOOTED
    assert event.count == 3


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("reserved_zero", 1, "reserved_zero"),
        ("last_damage", 15, "fixed damage"),
        ("host_slot_0", 40, "loot slot"),
        ("world_loot_mask", 128, "world loot mask"),
        ("host_extraction_progress", 106, "progress"),
    ),
)
def test_protocol_rejects_invalid_or_non_auditable_state(
    field: str, value: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(snapshot(), **{field: value}).validate()
