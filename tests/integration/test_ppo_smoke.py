import json
import math
import os
from pathlib import Path

import psutil
import pytest
import torch

from botcolosseo.cli.train_ppo import main


@pytest.mark.skipif(
    os.environ.get("BOTCOLOSSEO_RUN_PPO_SMOKE") != "1",
    reason="set BOTCOLOSSEO_RUN_PPO_SMOKE=1 for the 2,000-step real GPU gate",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_ppo_smoke_resumes_to_2000_steps(tmp_path: Path) -> None:
    output = tmp_path / "ppo-smoke"
    common = [
        "--device",
        "cuda:0",
        "--environment-steps",
        "2000",
        "--rollout-steps",
        "256",
        "--output-dir",
        str(output),
    ]

    assert main([*common, "--stop-after-steps", "1000"]) == 0
    first = json.loads((output / "summary.json").read_text())
    assert first["environment_steps"] == 1000
    assert first["completed"] is False
    assert main([*common, "--resume", str(output / "latest.pt")]) == 0
    final = json.loads((output / "summary.json").read_text())
    records = [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    training = [record for record in records if record["kind"] == "train"]

    assert final["environment_steps"] == 2000
    assert final["completed"] is True
    assert final["episode_count"] >= 1
    assert final["event_counts"]
    assert training and all(
        math.isfinite(record[name])
        for record in training
        for name in ("total_loss", "policy_loss", "value_loss", "entropy")
    )
    commands = [
        " ".join(process.info["cmdline"] or [])
        for process in psutil.process_iter(["cmdline"])
    ]
    assert not any("botcolosseo-duel" in command for command in commands)
    assert not any("vizdoom" in command.lower() for command in commands)
