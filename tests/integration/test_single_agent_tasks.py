from pathlib import Path

import pytest

from botcolosseo.envs.events import EventType
from botcolosseo.envs.task_runner import run_teacher_episode
from botcolosseo.scenarios.splits import TaskKind

EXPECTED_EVENT = {
    TaskKind.NAVIGATION: EventType.TASK_SUCCESS,
    TaskKind.PICKUP: EventType.PICKUP,
    TaskKind.RETURN: EventType.SCORE,
    TaskKind.STATIC_HIT: EventType.VALID_HIT,
    TaskKind.MOVING_HIT: EventType.VALID_HIT,
}
TEACHERS = {
    TaskKind.NAVIGATION: "fixed_route",
    TaskKind.PICKUP: "objective_first",
    TaskKind.RETURN: "evasive_return",
    TaskKind.STATIC_HIT: "aggressive_script",
    TaskKind.MOVING_HIT: "aggressive_script",
}


@pytest.mark.integration
@pytest.mark.timeout(30)
@pytest.mark.parametrize("task", tuple(TaskKind))
def test_teacher_completes_real_task_with_expected_event(task: TaskKind) -> None:
    summary = run_teacher_episode(task=task, teacher_name=TEACHERS[task], seed=17)

    assert summary.success, summary.to_json()
    assert not summary.truncated
    assert EXPECTED_EVENT[task].value in summary.event_types
    assert "task_success" in summary.event_types
    assert summary.first_frame_shape == (84, 84)


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_required_crystal_run_video_is_nonempty(tmp_path: Path) -> None:
    video_path = tmp_path / "m1-static-hit.mp4"

    summary = run_teacher_episode(
        task=TaskKind.STATIC_HIT,
        teacher_name="aggressive_script",
        seed=17,
        video_path=video_path,
        require_video=True,
    )

    assert summary.success
    assert summary.video_error is None
    assert video_path.stat().st_size > 0
