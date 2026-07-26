from __future__ import annotations

from botcolosseo.cli.evaluate_extraction_x0 import (
    _combat_cache_extract,
    _route_and_replacement,
)


def test_real_extraction_x0_mechanics_gate() -> None:
    route = _route_and_replacement(20260726)
    combat = _combat_cache_extract(20260727)

    assert route["host_banked"] == 100
    assert route["opponent_banked"] == 10
    assert route["event_counts"]["loot_drop"] == 2
    assert route["event_counts"]["extracted"] == 2
    assert combat["health_trace"] == [100, 80, 60, 40, 20, 0]
    assert combat["event_counts"]["valid_hit"] == 5
    assert combat["event_counts"]["death"] == 1
    assert combat["event_counts"]["cache_created"] == 1
    assert combat["event_counts"]["cache_looted"] >= 1
    assert combat["event_counts"]["extracted"] == 1
