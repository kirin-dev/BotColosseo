from botcolosseo.evaluation.extraction import DisengagementTracker


def test_disengagement_requires_live_distance_creation() -> None:
    tracker = DisengagementTracker()

    tracker.observe_encounter(health=40, ammo=12, opponent_distance=300)
    resolved = tracker.resolve_after_action(
        opponent_distance=520,
        learner_alive=True,
        opponent_alive=True,
    )

    assert resolved is True
    assert tracker.opportunities == 1
    assert tracker.successes == 1
    assert tracker.active is False


def test_opponent_death_is_not_a_successful_disengagement() -> None:
    tracker = DisengagementTracker()

    tracker.observe_encounter(health=35, ammo=10, opponent_distance=300)
    resolved = tracker.resolve_after_action(
        opponent_distance=900,
        learner_alive=True,
        opponent_alive=False,
    )

    assert resolved is False
    assert tracker.opportunities == 1
    assert tracker.successes == 0
    assert tracker.active is False


def test_learner_terminal_transition_cancels_disengagement() -> None:
    tracker = DisengagementTracker()

    tracker.observe_encounter(health=35, ammo=10, opponent_distance=300)
    resolved = tracker.resolve_after_action(
        opponent_distance=900,
        learner_alive=False,
        opponent_alive=True,
    )

    assert resolved is False
    assert tracker.successes == 0
    assert tracker.active is False
