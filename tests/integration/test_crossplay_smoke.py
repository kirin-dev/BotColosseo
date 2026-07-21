import multiprocessing as mp
import os
from pathlib import Path

import pytest
import torch

from botcolosseo.cli.smoke_crossplay import main


@pytest.mark.skipif(
    os.environ.get("BOTCOLOSSEO_RUN_M3_CROSSPLAY_SMOKE") != "1",
    reason="set BOTCOLOSSEO_RUN_M3_CROSSPLAY_SMOKE=1 after preparing the M3 smoke checkpoint",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_crossplay_smoke_runs_both_sides_and_cleans_workers() -> None:
    checkpoint = Path.cwd() / "runs/m3/smoke-input/base.pt"
    if not checkpoint.is_file():
        pytest.fail("prepare the hash-bound M3 smoke checkpoint before enabling this gate")
    before = {process.pid for process in mp.active_children()}

    assert (
        main(
            [
                "--checkpoint",
                str(checkpoint),
                "--device",
                os.environ.get("BOTCOLOSSEO_M3_CROSSPLAY_SMOKE_DEVICE", "cuda:1"),
                "--pairs",
                "1",
            ]
        )
        == 0
    )

    leaked = [process for process in mp.active_children() if process.pid not in before]
    assert leaked == []
