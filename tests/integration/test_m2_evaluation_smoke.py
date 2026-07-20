from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import psutil
import pytest

from botcolosseo.cli.evaluate_m2 import main


@pytest.mark.skipif(
    os.environ.get("BOTCOLOSSEO_RUN_M2_EVAL_SMOKE") != "1",
    reason="set BOTCOLOSSEO_RUN_M2_EVAL_SMOKE=1 for the real paired duel gate",
)
def test_real_random_and_script_evaluation_is_paired_and_cleans_up(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"

    result = main(
        [
            "--split",
            "validation",
            "--development",
            "--max-pairs",
            "1",
            "--max-decisions",
            "64",
            "--policies",
            "random_legal",
            "objective_first",
            "--opponents",
            "fixed_route",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    rows = list(csv.DictReader((output / "episodes.csv").open()))
    assert len(rows) == 4
    assert [(row["policy"], row["learner_side"]) for row in rows] == [
        ("random_legal", "host"),
        ("random_legal", "opponent"),
        ("objective_first", "host"),
        ("objective_first", "opponent"),
    ]
    assert len({row["pair_index"] for row in rows}) == 1
    assert all(int(row["peer_tic_lag_max"]) == 0 for row in rows)
    summary = json.loads((output / "summary.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert summary["official"] is False
    assert summary["passed"] is False
    assert manifest["official"] is False
    assert manifest["episodes_sha256"]
    assert manifest["summary_sha256"]
    commands = [
        " ".join(process.info["cmdline"] or [])
        for process in psutil.process_iter(["cmdline"])
    ]
    assert not any("botcolosseo-duel" in command for command in commands)
    assert not any("vizdoom" in command.lower() for command in commands)
