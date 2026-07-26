from __future__ import annotations

from pathlib import Path

import pytest
import torch

from botcolosseo.agents.checkpoint import (
    CheckpointMetadata,
    save_training_checkpoint,
)
from botcolosseo.agents.extraction_model import (
    create_extraction_actor,
    create_extraction_actor_critic,
)
from botcolosseo.training.extraction_checkpoint import (
    load_extraction_bc_warm_start,
    load_extraction_strong_actor,
    load_extraction_strong_actor_critic,
    sha256_file,
)


def save_checkpoint(path: Path, model: torch.nn.Module, scenario: str) -> Path:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    return save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=CheckpointMetadata("config", scenario, {"updates": 3}),
    )


def test_bc_warm_start_loads_actor_without_overwriting_critic(tmp_path: Path) -> None:
    actor = create_extraction_actor()
    with torch.no_grad():
        actor.policy.bias.fill_(0.25)
    checkpoint = save_checkpoint(tmp_path / "bc.pt", actor, "scenario")
    model = create_extraction_actor_critic()
    critic_before = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if not name.startswith("actor.")
    }

    metadata = load_extraction_bc_warm_start(
        checkpoint,
        model,
        expected_scenario_hash="scenario",
        expected_sha256=sha256_file(checkpoint),
    )

    assert metadata.counters["updates"] == 3
    assert torch.equal(model.actor.policy.bias, actor.policy.bias)
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in critic_before.items()
    )


def test_strong_actor_loader_rejects_identity_drift(tmp_path: Path) -> None:
    model = create_extraction_actor_critic()
    checkpoint = save_checkpoint(tmp_path / "ppo.pt", model, "scenario")

    actor, metadata = load_extraction_strong_actor(
        checkpoint,
        expected_scenario_hash="scenario",
        expected_sha256=sha256_file(checkpoint),
    )

    assert metadata.counters["updates"] == 3
    assert actor.scalar_dim == 9
    actor_critic, _ = load_extraction_strong_actor_critic(
        checkpoint,
        expected_scenario_hash="scenario",
    )
    assert actor_critic.actor.scalar_dim == 9
    with pytest.raises(ValueError, match="scenario"):
        load_extraction_strong_actor(
            checkpoint,
            expected_scenario_hash="changed",
        )
