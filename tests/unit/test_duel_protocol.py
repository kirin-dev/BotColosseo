from dataclasses import replace

import pytest

from botcolosseo.envs.duel_protocol import (
    DuelEventDecoder,
    DuelEventType,
    DuelProtocolSnapshot,
)


def snapshot(**changes: int) -> DuelProtocolSnapshot:
    base = DuelProtocolSnapshot(
        protocol_version=2,
        engine_tic=10,
        round_state=1,
        carrier=0,
        core_state=0,
        host_pickups=0,
        opponent_pickups=0,
        host_drops=0,
        opponent_drops=0,
        host_scores=0,
        opponent_scores=0,
        host_valid_hits=0,
        opponent_valid_hits=0,
        host_deaths=0,
        opponent_deaths=0,
        host_respawns=0,
        opponent_respawns=0,
        host_score=0,
        opponent_score=0,
        winner=0,
        core_returns=0,
        core_x=0,
        core_y=0,
        spawn_index=0,
        reserved_zero=0,
    )
    return replace(base, **changes)


def test_snapshot_requires_exact_version_ranges_and_reserved_zero() -> None:
    values = list(snapshot().to_values())

    assert DuelProtocolSnapshot.from_values(values) == snapshot()
    for index, value in ((0, 1), (3, 3), (19, 3), (24, 1)):
        invalid = values.copy()
        invalid[index] = value
        with pytest.raises(ValueError):
            DuelProtocolSnapshot.from_values(invalid)


def test_decoder_emits_stable_side_specific_events_once() -> None:
    decoder = DuelEventDecoder()
    decoder.reset(snapshot())

    events = decoder.decode(
        snapshot(host_pickups=1, opponent_valid_hits=1),
        episode_id=4,
        decision_index=2,
    )

    assert [(event.side, event.type) for event in events] == [
        ("host", DuelEventType.PICKUP),
        ("opponent", DuelEventType.VALID_HIT),
    ]
    assert decoder.decode(
        snapshot(host_pickups=1, opponent_valid_hits=1),
        episode_id=4,
        decision_index=3,
    ) == ()


@pytest.mark.parametrize(
    "current",
    (snapshot(host_pickups=2), snapshot(engine_tic=9), snapshot(protocol_version=1)),
)
def test_decoder_rejects_counter_jumps_time_reversal_and_version(current) -> None:
    decoder = DuelEventDecoder()
    decoder.reset(snapshot())

    with pytest.raises(ValueError):
        decoder.decode(current, episode_id=0, decision_index=1)
