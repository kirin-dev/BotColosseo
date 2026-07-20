import pytest

from botcolosseo.envs.task_runner import run_teacher_episode
from botcolosseo.scenarios.splits import TaskKind


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_same_seed_teacher_replays_terminal_and_event_sequence() -> None:
    first = run_teacher_episode(
        task=TaskKind.NAVIGATION,
        teacher_name="fixed_route",
        seed=23,
    )
    second = run_teacher_episode(
        task=TaskKind.NAVIGATION,
        teacher_name="fixed_route",
        seed=23,
    )

    assert first.success == second.success
    assert first.truncated == second.truncated
    assert first.decisions == second.decisions
    assert first.event_types == second.event_types
