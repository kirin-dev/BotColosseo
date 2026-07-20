import subprocess
import sys
from pathlib import Path

import pytest
import vizdoom as vzd

from botcolosseo.envs.smoke import run_smoke
from botcolosseo.envs.vizdoom_game import GameSettings


@pytest.mark.integration
def test_basic_scenario_terminates_and_resets() -> None:
    settings = GameSettings(config_path=Path(vzd.scenarios_path) / "basic.cfg", seed=17)

    summary = run_smoke(settings, episodes=2, max_decisions=100, frame_skip=4)

    assert summary.all_terminated
    assert len(summary.episodes) == 2
    assert all(item.first_frame_shape for item in summary.episodes)


@pytest.mark.integration
def test_basic_scenario_records_mp4(tmp_path: Path) -> None:
    settings = GameSettings(config_path=Path(vzd.scenarios_path) / "basic.cfg", seed=17)
    video_path = tmp_path / "smoke.mp4"

    summary = run_smoke(
        settings,
        episodes=1,
        max_decisions=100,
        frame_skip=4,
        video_path=video_path,
        require_video=True,
    )

    assert summary.all_terminated
    assert summary.video_error is None
    assert video_path.stat().st_size > 0


@pytest.mark.integration
def test_smoke_cli_returns_success() -> None:
    repository_root = Path(__file__).parents[2]

    result = subprocess.run(
        [sys.executable, "scripts/smoke_vizdoom.py", "--episodes", "1"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"all_terminated": true' in result.stdout
