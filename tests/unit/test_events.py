import json

import pytest

from botcolosseo.envs.events import (
    EventDecoder,
    EventProtocolError,
    EventType,
    ProtocolSnapshot,
)


def snapshot(**overrides: int) -> ProtocolSnapshot:
    values = [1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    indices = {name: index for index, name in enumerate(ProtocolSnapshot.field_names())}
    for name, value in overrides.items():
        values[indices[name]] = value
    return ProtocolSnapshot.from_values(values)


def test_pickup_delta_emits_once_and_serializes() -> None:
    decoder = EventDecoder()
    decoder.decode(snapshot(), region_name="home", episode_id=3, decision_index=0)

    events = decoder.decode(
        snapshot(engine_tic=14, pickup_count=1),
        region_name="lower_route",
        episode_id=3,
        decision_index=1,
    )

    assert [event.type for event in events] == [EventType.PICKUP, EventType.REGION_TRANSITION]
    assert json.loads(events[0].to_json())["type"] == "pickup"
    assert decoder.decode(
        snapshot(engine_tic=18, pickup_count=1),
        region_name="lower_route",
        episode_id=3,
        decision_index=2,
    ) == ()


def test_simultaneous_counters_have_stable_order() -> None:
    decoder = EventDecoder()
    decoder.decode(snapshot(), region_name="center", episode_id=0, decision_index=0)

    events = decoder.decode(
        snapshot(engine_tic=14, pickup_count=1, score_count=1),
        region_name="center",
        episode_id=0,
        decision_index=1,
    )

    assert [event.type for event in events] == [EventType.PICKUP, EventType.SCORE]


@pytest.mark.parametrize(
    ("previous", "current", "message"),
    [
        ({"pickup_count": 1}, {"pickup_count": 0}, "decreased"),
        ({}, {"pickup_count": 2}, "jumped"),
        ({}, {"protocol_version": 2}, "protocol"),
        ({}, {"reserved_zero": 1}, "reserved"),
    ],
)
def test_invalid_protocol_snapshots_are_rejected(previous, current, message: str) -> None:
    decoder = EventDecoder()
    decoder.decode(snapshot(**previous), region_name="home", episode_id=0, decision_index=0)

    with pytest.raises(EventProtocolError, match=message):
        decoder.decode(
            snapshot(engine_tic=14, **current),
            region_name="home",
            episode_id=0,
            decision_index=1,
        )


def test_reset_allows_episode_counters_to_restart() -> None:
    decoder = EventDecoder()
    decoder.decode(snapshot(pickup_count=1), region_name="center", episode_id=0, decision_index=0)

    decoder.reset()

    assert decoder.decode(
        snapshot(pickup_count=0),
        region_name="home",
        episode_id=1,
        decision_index=0,
    ) == ()
