import json
from pathlib import Path

import pytest
import torch

from botcolosseo.agents.checkpoint import (
    CheckpointMetadata,
    save_training_checkpoint,
)
from botcolosseo.cli.train_extraction_style import (
    _initialize_style_weights,
    _resolved_opportunity_config,
    _resolved_style_reward_config,
    build_parser,
)
from botcolosseo.data.demonstrations import sha256_file


def _parent_checkpoint(root: Path) -> tuple[Path, torch.nn.Module]:
    output = root / "runs/extraction/styles/aggressive"
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    checkpoint = save_training_checkpoint(
        output / "candidate-0600000.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=CheckpointMetadata(
            "parent-config",
            "scenario",
            {"environment_steps": 600_000, "episodes": 10, "updates": 20},
        ),
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "base_checkpoint_sha256": "base-sha",
                "checkpoint_sha256": sha256_file(checkpoint),
                "completed": True,
                "config_hash": "parent-config",
                "environment_steps": 600_000,
                "scenario_hash": "scenario",
                "style": "aggressive",
                "test_cases_accessed": False,
                "updates": 20,
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, model


def test_style_cli_rejects_resume_with_weights_only_initialization() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--style",
                "aggressive",
                "--resume",
                "latest.pt",
                "--initialize-from",
                "parent.pt",
            ]
        )


def test_style_weights_only_initialization_validates_and_records_parent(
    tmp_path: Path,
) -> None:
    checkpoint, source = _parent_checkpoint(tmp_path)
    target = torch.nn.Linear(2, 2)

    lineage = _initialize_style_weights(
        checkpoint=checkpoint,
        model=target,
        style="aggressive",
        base_checkpoint_sha256="base-sha",
        scenario_hash="scenario",
        root=tmp_path,
    )

    assert lineage == {
        "initialization_mode": "weights_only",
        "parent_checkpoint": (
            "runs/extraction/styles/aggressive/candidate-0600000.pt"
        ),
        "parent_checkpoint_sha256": sha256_file(checkpoint),
        "parent_config_hash": "parent-config",
        "parent_environment_steps": 600_000,
        "parent_summary": "runs/extraction/styles/aggressive/summary.json",
        "parent_updates": 20,
    }
    assert all(
        torch.equal(value, source.state_dict()[name])
        for name, value in target.state_dict().items()
    )


def test_style_weights_only_initialization_rejects_base_drift(
    tmp_path: Path,
) -> None:
    checkpoint, _ = _parent_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="provenance"):
        _initialize_style_weights(
            checkpoint=checkpoint,
            model=torch.nn.Linear(2, 2),
            style="aggressive",
            base_checkpoint_sha256="different-base",
            scenario_hash="scenario",
            root=tmp_path,
        )


def test_defensive_reward_overrides_are_targeted_and_auditable() -> None:
    config = _resolved_style_reward_config(
        "defensive",
        {
            "risk_disengagement": 0.30,
            "combat_with_value": -0.030,
        },
    )

    assert config.risk_disengagement == pytest.approx(0.30)
    assert config.combat_with_value == pytest.approx(-0.030)
    assert config.meaningful_extraction == pytest.approx(0.20)
    assert config.empty_idle == pytest.approx(-0.003)


@pytest.mark.parametrize(
    "overrides",
    (
        {"unknown": 1.0},
        {"risk_disengagement": float("nan")},
        {"risk_disengagement": True},
        {"risk_disengagement_cap": 1.5},
    ),
)
def test_style_reward_overrides_fail_closed(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="override"):
        _resolved_style_reward_config("defensive", overrides)


def test_opportunity_overrides_are_style_specific_and_validated() -> None:
    aggressive = _resolved_opportunity_config(
        "aggressive",
        {"completion_utility": 0.4, "maximum_carried_value": 50},
    )

    assert aggressive.completion_utility == pytest.approx(0.4)
    assert aggressive.maximum_carried_value == 50
    assert aggressive.attack_distance == pytest.approx(512)

    with pytest.raises(ValueError, match="override"):
        _resolved_opportunity_config("aggressive", {"risk_distance": 512})
    with pytest.raises(ValueError, match="configuration"):
        _resolved_opportunity_config(
            "aggressive", {"attack_distance": 900, "engagement_distance": 700}
        )
