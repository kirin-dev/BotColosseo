import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import pytest

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.synchronous_duel import SynchronousDuelEnv
from botcolosseo.scenarios.regions import RegionGraph


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_real_duel_stays_synchronized_and_cleans_workers() -> None:
    before = {child.pid for child in mp.active_children()}
    graph = RegionGraph.from_yaml(Path("assets/scenarios/crystal_run/src/regions.yaml"))
    env = SynchronousDuelEnv(
        config_path=Path("assets/scenarios/crystal_run/crystal_run.cfg"),
        region_graph=graph,
        seed=17,
    )
    try:
        observations, info = env.reset()
        assert observations.host.frame.shape == (84, 84)
        assert observations.opponent.frame.shape == (84, 84)
        assert info.protocol_version == 2
        last_tic = info.engine_tic
        for _ in range(50):
            step = env.step(MacroAction.IDLE, MacroAction.IDLE)
            assert step.engine_tic == last_tic + 4
            last_tic = step.engine_tic
            if step.terminated or step.truncated:
                break
        _, reset_info = env.reset()
        assert reset_info.episode_id == 1
        last_tic = reset_info.engine_tic
        for _ in range(10):
            step = env.step(MacroAction.IDLE, MacroAction.IDLE)
            assert step.engine_tic == last_tic + 4
            last_tic = step.engine_tic
    finally:
        env.close()

    assert {child.pid for child in mp.active_children()} <= before


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_real_duel_smoke_cli_reports_cleanup() -> None:
    root = Path(__file__).parents[2]

    result = subprocess.run(
        [sys.executable, "scripts/smoke_duel.py", "--decisions", "10", "--seed", "17"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"completed_decisions": 10' in result.stdout
    assert '"cleaned_workers": true' in result.stdout
