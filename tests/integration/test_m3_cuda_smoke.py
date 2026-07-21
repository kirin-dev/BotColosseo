import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import pytest
import torch

from botcolosseo.agents.league_opponents import (
    CheckpointOpponentPolicy,
    OpponentSpec,
    sha256_file,
)
from botcolosseo.cli.train_league import main


@pytest.mark.skipif(
    os.environ.get("BOTCOLOSSEO_RUN_M3_CUDA_SMOKE") != "1",
    reason="set BOTCOLOSSEO_RUN_M3_CUDA_SMOKE=1 after preparing runs/m3/smoke-input",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.timeout(180)
def test_m3_cuda_train_update_checkpoint_reload_and_cleanup(tmp_path: Path) -> None:
    root = Path.cwd()
    inputs = root / "runs/m3/smoke-input"
    required = tuple(inputs / name for name in ("base.pt", "pool.json", "payoffs.json"))
    if not all(path.is_file() for path in required):
        pytest.fail("prepare the hash-bound M3 smoke inputs before enabling this gate")
    output = tmp_path / "m3-cuda-smoke"
    before = {process.pid for process in mp.active_children()}
    device_name = os.environ.get("BOTCOLOSSEO_M3_CUDA_SMOKE_DEVICE", "cuda:1")

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
                device_name,
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
    training = [row for row in records if row["kind"] == "train"]
    candidate = output / summary["candidate_checkpoints"][-1]["checkpoint"]
    candidate_hash = sha256_file(candidate)
    assert candidate_hash == summary["candidate_checkpoints"][-1]["sha256"]
    spec = OpponentSpec(
        opponent_id="cuda-smoke-candidate",
        kind="checkpoint",
        checkpoint=str(candidate),
        checkpoint_sha256=candidate_hash,
        scenario_hash=summary["scenario_hash"],
        selection_evidence="integration:cuda-smoke",
    )
    reloaded = CheckpointOpponentPolicy.load(spec, device=torch.device(device_name))

    assert summary["completed"] is True
    assert summary["environment_steps"] == 2048
    assert summary["event_counts"] and summary["opponent_source_counts"]
    assert training and all(
        math.isfinite(row[name])
        for row in training
        for name in ("total_loss", "policy_loss", "value_loss", "entropy")
    )
    assert all(not parameter.requires_grad for parameter in reloaded._actor.parameters())
    assert {process.pid for process in mp.active_children()} <= before
