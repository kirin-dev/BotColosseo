from __future__ import annotations

import json
from pathlib import Path

import pytest

from botcolosseo.demo.m2_training_plot import load_training_evidence


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_training_plot_loader_returns_validation_only_curves(tmp_path: Path) -> None:
    bc = _write(
        tmp_path / "bc.json",
        {
            "official_test_result": False,
            "test_cases_accessed": False,
            "selection": {
                "selected_update": 250,
                "validation_curve": [
                    {"update": 250, "loss": 0.2, "accuracy": 0.9}
                ],
            },
        },
    )
    ppo = _write(
        tmp_path / "ppo.json",
        {
            "official_test_result": False,
            "test_cases_accessed": False,
            "selected": {"environment_steps": 100_000},
            "selection": {
                "split": "validation",
                "candidates": [
                    {
                        "environment_steps": 100_000,
                        "objective_rate": 1.0,
                        "win_rate": 0.8,
                    }
                ],
            },
        },
    )

    evidence = load_training_evidence(bc, ppo)

    assert evidence.bc_selected_update == 250
    assert evidence.ppo_selected_steps == 100_000
    assert evidence.ppo_win_rates == (0.8,)


def test_training_plot_loader_rejects_test_access(tmp_path: Path) -> None:
    payload = {
        "official_test_result": False,
        "test_cases_accessed": True,
        "selection": {"selected_update": 1, "validation_curve": []},
    }
    bc = _write(tmp_path / "bc.json", payload)
    ppo = _write(tmp_path / "ppo.json", payload)

    with pytest.raises(ValueError, match="validation-only"):
        load_training_evidence(bc, ppo)
