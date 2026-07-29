from __future__ import annotations

from pathlib import Path

import yaml

from botcolosseo.cli.summarize_extraction_ablations import (
    COEFFICIENTS,
    STYLES,
    VARIANTS,
)


def _config(name: str) -> dict[str, object]:
    return yaml.safe_load(
        Path(f"configs/extraction/ablations/{name}.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_ablation_matrix_removes_only_declared_regularizers() -> None:
    assert STYLES == ("aggressive", "defensive", "explorer")
    assert VARIANTS == ("full", "reward-plus-kl", "reward-only")
    assert COEFFICIENTS == {
        "full": (0.08, 0.01),
        "reward-plus-kl": (0.08, 0.00),
        "reward-only": (0.00, 0.00),
    }

    full = yaml.safe_load(
        Path("configs/extraction/styles.yaml").read_text(encoding="utf-8")
    )
    ignored = {"beta_kl", "rho_residual", "output_root"}
    for variant in ("reward-plus-kl", "reward-only"):
        ablation = _config(variant)
        assert {
            key: value for key, value in ablation.items() if key not in ignored
        } == {
            key: value for key, value in full.items() if key not in ignored
        }
        assert (
            float(ablation["beta_kl"]),
            float(ablation["rho_residual"]),
        ) == COEFFICIENTS[variant]


def test_defensive_ablations_preserve_calibrated_reward_and_horizon() -> None:
    full = yaml.safe_load(
        Path(
            "configs/extraction/styles-defensive-calibration.yaml"
        ).read_text(encoding="utf-8")
    )
    ignored = {"beta_kl", "rho_residual", "output_root"}
    for variant in ("reward-plus-kl", "reward-only"):
        ablation = _config(f"{variant}-defensive")
        assert {
            key: value for key, value in ablation.items() if key not in ignored
        } == {
            key: value for key, value in full.items() if key not in ignored
        }
        assert ablation["environment_steps"] == 400_000
        assert ablation["style_reward_overrides"]["defensive"] == {
            "risk_disengagement": 0.30,
            "combat_with_value": -0.030,
        }


def test_ablation_runner_has_two_fixed_gpu_lanes_and_200k_stop() -> None:
    source = Path("scripts/run_extraction_v3_ablations.sh").read_text(
        encoding="utf-8"
    )

    assert "TARGET_STEPS=200000" in source
    assert "run_lane 0" in source
    assert "run_lane 1" in source
    assert "reward-only aggressive" in source
    assert "reward-plus-kl explorer" in source
    assert "BOTCOLOSSEO_ABLATION_PREFLIGHT_ONLY" in source
    assert "--split validation" in source
    assert "--stop-after-steps \"$TARGET_STEPS\"" in source
    assert "--split heldout" not in source
    assert "--split test" not in source
