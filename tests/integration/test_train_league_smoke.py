import json
import math
import os
from pathlib import Path

import psutil
import pytest
import torch

from botcolosseo.cli.train_league import main


@pytest.mark.skipif(
    os.environ.get("BOTCOLOSSEO_RUN_M3_SMOKE") != "1",
    reason="set BOTCOLOSSEO_RUN_M3_SMOKE=1 after preparing runs/m3/smoke-input",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_league_smoke_trains_2048_steps_and_cleans_workers(
    tmp_path: Path,
) -> None:
    root = Path.cwd()
    inputs = root / "runs/m3/smoke-input"
    required = tuple(inputs / name for name in ("base.pt", "pool.json", "payoffs.json"))
    if not all(path.is_file() for path in required):
        pytest.fail("prepare the hash-bound M3 smoke inputs before enabling this gate")
    output = tmp_path / "league-smoke"
    device = os.environ.get("BOTCOLOSSEO_M3_SMOKE_DEVICE", "cuda:1")

    assert (
        main(
            [
                "--base-checkpoint",
                str(required[0]),
                "--pool",
                str(required[1]),
                "--payoffs",
                str(required[2]),
                "--run-dir",
                str(output),
                "--device",
                device,
                "--allow-provisional-base",
                "--environment-steps",
                "2048",
                "--rollout-steps",
                "256",
            ]
        )
        == 0
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    training = [record for record in records if record["kind"] == "train"]

    assert summary["completed"] is True
    assert summary["environment_steps"] == 2048
    assert summary["candidate_checkpoints"]
    assert summary["event_counts"] and summary["opponent_source_counts"]
    assert training and all(
        math.isfinite(record[name])
        for record in training
        for name in ("total_loss", "policy_loss", "value_loss", "entropy")
    )
    commands = [
        " ".join(process.info["cmdline"] or [])
        for process in psutil.process_iter(["cmdline"])
    ]
    assert not any(str(output) in command for command in commands)
