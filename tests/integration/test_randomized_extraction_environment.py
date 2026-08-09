from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_layouts import randomized_layout_variant
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_randomized_extraction_reset_step_and_cleanup() -> None:
    before = {child.pid for child in mp.active_children()}
    seed = 62000
    root = Path(__file__).resolve().parents[2]
    env = SynchronousExtractionEnv(
        config_path=root
        / "assets/scenarios/crystal_run_extraction_randomized/"
        "crystal_run_extraction_randomized.cfg",
        seed=seed,
        max_decisions=20,
        layout_variant=randomized_layout_variant(seed),
    )
    try:
        observations, info = env.reset()
        assert observations.host.frame.shape == (84, 84)
        assert observations.opponent.frame.shape == (84, 84)
        assert info.protocol_version == 3
        assert env.protocol_snapshot().world_loot_mask == 127
        last_tic = info.engine_tic
        for _ in range(5):
            step = env.step(MacroAction.IDLE, MacroAction.IDLE)
            assert step.engine_tic == last_tic + 4
            assert step.peer_tic_lag <= 2
            last_tic = step.engine_tic
    finally:
        env.close()

    assert {child.pid for child in mp.active_children()} <= before
