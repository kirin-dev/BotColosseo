from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import botcolosseo.cli.evaluate_hybrid_difficulty as cli


def test_hybrid_difficulty_preflight_freezes_200_validation_episodes(
    capsys,
    monkeypatch,
) -> None:
    root = Path(__file__).resolve().parents[2]
    scenario_hash = "9" * 64
    source = SimpleNamespace(governor_config=root / "unused.yaml")
    product = SimpleNamespace(
        defensive=source,
        explorer=source,
        scenario_hash=scenario_hash,
        difficulty_config=root / "configs/difficulty.yaml",
        cases=root / "configs/m2/validation.json",
        config_sha256="1" * 64,
        difficulty_config_sha256="2" * 64,
        cases_sha256="3" * 64,
    )
    hybrid = SimpleNamespace(
        style="defensive",
        scenario_hash=scenario_hash,
        config_sha256="4" * 64,
        base_checkpoint_sha256="5" * 64,
    )
    monkeypatch.setattr(
        cli,
        "load_hybrid_difficulty_product_config",
        lambda *args, **kwargs: product,
    )
    monkeypatch.setattr(
        cli,
        "load_hybrid_policy_config",
        lambda *args, **kwargs: hybrid,
    )

    assert (
        cli.main(
            [
                "--style",
                "defensive",
                "--output-dir",
                "unused",
                "--preflight",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "m5-hybrid-defensive-difficulty-extension"
    assert payload["difficulties"] == ["easy", "normal"]
    assert payload["expected_episodes"] == 200
    assert payload["test_cases_accessed"] is False
    assert payload["preflight_passed"] is True
