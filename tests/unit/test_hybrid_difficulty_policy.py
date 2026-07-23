from __future__ import annotations

from collections.abc import Sequence

from botcolosseo.agents.difficulty import DifficultyProfile
from botcolosseo.agents.hybrid_difficulty import TracedHybridDifficultyPolicy
from botcolosseo.agents.style_governor import GovernorTelemetry
from botcolosseo.envs.actions import MacroAction


class StubHybrid:
    def __init__(
        self,
        rows: Sequence[GovernorTelemetry],
    ) -> None:
        self._rows = iter(rows)
        self._telemetry: list[GovernorTelemetry] = []

    @property
    def telemetry(self) -> tuple[GovernorTelemetry, ...]:
        return tuple(self._telemetry)

    def reset(self) -> None:
        self._telemetry.clear()

    def act(self, observation: object) -> MacroAction:
        del observation
        row = next(self._rows)
        self._telemetry.append(row)
        return row.final_action

    def drain_telemetry(self) -> tuple[GovernorTelemetry, ...]:
        rows = self.telemetry
        self._telemetry.clear()
        return rows


def _row(index: int, action: MacroAction, *, intervened: bool) -> GovernorTelemetry:
    return GovernorTelemetry(
        decision_index=index,
        base_action=MacroAction.IDLE,
        final_action=action,
        state="guard",
        trigger="score_rise",
        reason="guard",
        intervened=intervened,
        used_override=False,
        fallback_condition="timeout",
        route_mode=None,
    )


def test_traced_hybrid_difficulty_matches_easy_hold_and_delay() -> None:
    hybrid = StubHybrid(
        (
            _row(0, MacroAction.MOVE_FORWARD, intervened=True),
            _row(1, MacroAction.ATTACK, intervened=False),
        )
    )
    policy = TracedHybridDifficultyPolicy(
        "defensive",
        hybrid,  # type: ignore[arg-type]
        DifficultyProfile(reaction_delay=2, policy_update_interval=2),
    )
    policy.reset(seed=3)

    actions = [policy.act(object(), object()) for _ in range(4)]

    assert actions == [
        MacroAction.IDLE,
        MacroAction.IDLE,
        MacroAction.MOVE_FORWARD,
        MacroAction.MOVE_FORWARD,
    ]
    assert [row.policy_updated for row in policy.trace] == [
        True,
        False,
        True,
        False,
    ]
    assert [row.warmup for row in policy.trace] == [True, True, False, False]
    assert [row.source_decision_index for row in policy.trace] == [
        None,
        None,
        0,
        0,
    ]
    assert all(row.intervened for row in policy.trace[2:])


def test_traced_hybrid_difficulty_reset_and_drain_are_episode_local() -> None:
    hybrid = StubHybrid(
        (
            _row(0, MacroAction.ATTACK, intervened=True),
            _row(0, MacroAction.MOVE_FORWARD, intervened=False),
        )
    )
    policy = TracedHybridDifficultyPolicy(
        "defensive",
        hybrid,  # type: ignore[arg-type]
        DifficultyProfile(reaction_delay=0, policy_update_interval=1),
    )
    policy.reset(seed=1)
    assert policy.act(object(), object()) is MacroAction.ATTACK
    telemetry, trace = policy.drain_evidence()
    assert len(telemetry) == len(trace) == 1
    assert not policy.trace

    policy.reset(seed=2)
    assert policy.act(object(), object()) is MacroAction.MOVE_FORWARD
    assert policy.trace[0].source_decision_index == 0
