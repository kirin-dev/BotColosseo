from __future__ import annotations

import dataclasses

import pytest

from botcolosseo.agents.style_governor import (
    ACTION_COUNT,
    DefensiveGovernor,
    DefensiveGovernorConfig,
    ExplorerGovernor,
    ExplorerGovernorConfig,
    PublicStyleContext,
)
from botcolosseo.envs.actions import MacroAction


def _context(**overrides: object) -> PublicStyleContext:
    values: dict[str, object] = {
        "health": 100.0,
        "armor": 0.0,
        "ammo": 20.0,
        "own_score": 0,
        "opponent_score": 0,
        "has_core": False,
        "previous_action": MacroAction.MOVE_FORWARD,
        "base_logits": (0.0,) * ACTION_COUNT,
        "decision_index": 0,
    }
    values.update(overrides)
    return PublicStyleContext(**values)  # type: ignore[arg-type]


def _defensive() -> DefensiveGovernor:
    return DefensiveGovernor(
        DefensiveGovernorConfig(
            guard_decisions=2,
            guard_bias=1.0,
            disengage_decisions=2,
            disengage_bias=1.5,
            recover_decisions=1,
            low_health_threshold=30.0,
            health_drop_threshold=20.0,
            max_consecutive_interventions=2,
        )
    )


def _explorer(*, stall_repeat_decisions: int = 8) -> ExplorerGovernor:
    return ExplorerGovernor(
        ExplorerGovernorConfig(
            route_decisions=3,
            route_bias=1.0,
            flank_bias=1.5,
            stall_repeat_decisions=stall_repeat_decisions,
            stall_recovery_decisions=1,
            low_health_threshold=25.0,
            max_consecutive_interventions=3,
        )
    )


def test_public_context_contains_no_privileged_or_case_fields() -> None:
    fields = {field.name for field in dataclasses.fields(PublicStyleContext)}

    assert fields.isdisjoint(
        {
            "seed",
            "case",
            "learner_side",
            "x",
            "y",
            "region",
            "carrier",
            "opponent_health",
            "teacher_state",
        }
    )


def test_public_context_rejects_nonfinite_or_wrong_sized_logits() -> None:
    with pytest.raises(ValueError, match="finite"):
        _context(base_logits=(0.0,) * (ACTION_COUNT - 1))
    with pytest.raises(ValueError, match="finite"):
        _context(base_logits=(float("nan"),) + (0.0,) * (ACTION_COUNT - 1))


def test_defensive_score_rise_runs_bounded_alternating_guard_then_base_recovery() -> None:
    governor = _defensive()
    governor.reset()
    governor.decide(_context())

    left = governor.decide(_context(own_score=1, decision_index=1))
    right = governor.decide(_context(own_score=1, decision_index=2))
    recovery = governor.decide(_context(own_score=1, decision_index=3))
    base = governor.decide(_context(own_score=1, decision_index=4))

    assert left.state == right.state == "guard"
    assert left.reason == "guard_scan_left"
    assert right.reason == "guard_scan_right"
    assert left.intervened and right.intervened
    assert recovery.state == "recover" and not recovery.intervened
    assert base.state == "base" and not base.intervened


def test_defensive_health_drop_disengages_and_carrying_forces_exact_base() -> None:
    governor = _defensive()
    governor.reset()
    governor.decide(_context())

    disengage = governor.decide(_context(health=75.0, decision_index=1))
    carrying = governor.decide(
        _context(health=75.0, has_core=True, decision_index=2)
    )

    assert disengage.state == "disengage"
    assert disengage.trigger == "health_drop"
    assert disengage.logit_bias[MacroAction.MOVE_BACKWARD] > 0.0
    assert carrying.state == "base"
    assert carrying.reason == "objective_return_priority"
    assert not carrying.intervened


def test_defensive_is_deterministic_for_identical_public_trajectory() -> None:
    trajectory = (
        _context(),
        _context(own_score=1, decision_index=1),
        _context(own_score=1, decision_index=2),
        _context(own_score=1, decision_index=3),
    )
    first = _defensive()
    second = _defensive()
    first.reset()
    second.reset()

    assert [first.decide(row) for row in trajectory] == [
        second.decide(row) for row in trajectory
    ]


@pytest.mark.parametrize(
    ("episode", "expected"),
    ((0, "upper"), (1, "lower"), (2, "flank"), (3, "upper")),
)
def test_explorer_episode_modes_cycle_without_seed(episode: int, expected: str) -> None:
    governor = _explorer()
    for _ in range(episode + 1):
        governor.reset()

    assert governor.episode_mode == expected


def test_explorer_leaves_pickup_to_base_then_commits_only_on_core_rise() -> None:
    governor = _explorer()
    governor.reset()
    warmup = governor.decide(_context())
    base = governor.decide(_context(decision_index=1))
    commit = governor.decide(_context(has_core=True, decision_index=2))

    assert not warmup.intervened and not base.intervened
    assert commit.state == "route_commit"
    assert commit.route_mode == "upper"
    assert commit.intervened
    assert commit.logit_bias[MacroAction.FORWARD_TURN_LEFT] > 0.0


def test_explorer_modes_have_distinct_mirrored_action_signatures() -> None:
    signatures: list[tuple[float, ...]] = []
    for episode in range(3):
        governor = _explorer()
        for _ in range(episode + 1):
            governor.reset()
        governor.decide(_context())
        signatures.append(governor.decide(_context(has_core=True)).logit_bias)

    upper, lower, flank = signatures
    assert upper[MacroAction.FORWARD_TURN_LEFT] > 0.0
    assert lower[MacroAction.FORWARD_TURN_RIGHT] > 0.0
    assert flank[MacroAction.STRAFE_LEFT] > upper[MacroAction.STRAFE_LEFT]
    assert len(set(signatures)) == 3


def test_explorer_score_drop_health_and_stall_use_exact_base_recovery() -> None:
    for trigger_context, expected_trigger in (
        (_context(has_core=False, decision_index=2), "core_drop"),
        (_context(has_core=True, health=20.0, decision_index=2), "health_safety"),
        (_context(has_core=True, own_score=1, decision_index=2), "own_score_rise"),
    ):
        governor = _explorer()
        governor.reset()
        governor.decide(_context())
        governor.decide(_context(has_core=True, decision_index=1))
        recovery = governor.decide(trigger_context)
        assert recovery.state == "stall_recovery"
        assert recovery.trigger == expected_trigger
        assert not recovery.intervened

    stalled = _explorer(stall_repeat_decisions=2)
    stalled.reset()
    stalled.decide(_context(previous_action=MacroAction.MOVE_FORWARD))
    stalled.decide(
        _context(
            has_core=True,
            previous_action=MacroAction.MOVE_FORWARD,
            decision_index=1,
        )
    )
    recovery = stalled.decide(
        _context(
            has_core=True,
            previous_action=MacroAction.MOVE_FORWARD,
            decision_index=2,
        )
    )
    assert recovery.trigger == "repeated_action_stall"
    assert not recovery.intervened


def test_governor_configs_reject_unbounded_intervention() -> None:
    with pytest.raises(ValueError, match="consecutive"):
        DefensiveGovernorConfig(3, 1.0, 2, 1.0, 1, 30.0, 20.0, 2)
    with pytest.raises(ValueError, match="consecutive"):
        ExplorerGovernorConfig(4, 1.0, 1.5, 3, 1, 25.0, 3)
