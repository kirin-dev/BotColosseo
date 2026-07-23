from botcolosseo.data.explorer_demonstrations import (
    ExplorerStepEvidence,
    label_explorer_windows,
)
from botcolosseo.envs.duel_protocol import DuelEvent, DuelEventType


def _event(side: str, event_type: DuelEventType, index: int) -> DuelEvent:
    return DuelEvent(event_type, side, 0, index, 4 * (index + 1))


def _step(
    route: str,
    region: str | None,
    *events: DuelEvent,
) -> ExplorerStepEvidence:
    return ExplorerStepEvidence(route, region, tuple(events))


def test_explorer_keeps_only_scored_matching_route_windows() -> None:
    steps = (
        _step("direct_upper", "home"),
        _step("direct_upper", "upper_route"),
        _step(
            "direct_upper",
            "home",
            _event("host", DuelEventType.SCORE, 2),
        ),
        _step("direct_lower", "lower_route"),
        _step(
            "direct_lower",
            "center",
            _event("opponent", DuelEventType.SCORE, 4),
        ),
    )

    labels = label_explorer_windows(steps, learner_side="host")

    assert labels.selected == (True, True, True, False, False)
    assert labels.reasons == (
        "successful_direct_upper",
        "successful_direct_upper",
        "successful_direct_upper",
        "rejected_route_window",
        "rejected_route_window",
    )
    assert labels.successful_windows == 1
    assert dict(labels.route_windows) == {
        "direct_upper": 1,
        "direct_lower": 0,
        "flank": 0,
    }


def test_explorer_rejects_mixed_or_incomplete_paths() -> None:
    mixed = (
        _step("direct_upper", "upper_route"),
        _step("direct_upper", "flank_west"),
        _step("direct_upper", "home", _event("host", DuelEventType.SCORE, 2)),
    )
    incomplete = (
        _step("direct_lower", "lower_route"),
        _step("direct_lower", "center"),
    )

    mixed_labels = label_explorer_windows(mixed, learner_side="host")
    incomplete_labels = label_explorer_windows(incomplete, learner_side="host")

    assert not any(mixed_labels.selected)
    assert not any(incomplete_labels.selected)
    assert mixed_labels.successful_windows == 0
    assert incomplete_labels.successful_windows == 0
