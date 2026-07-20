from pathlib import Path

import numpy as np
import pytest
import torch

from botcolosseo.agents.checkpoint import (
    CheckpointMetadata,
    load_training_checkpoint,
    save_training_checkpoint,
)


def update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    features = torch.randn(8, 4, device=device)
    target = torch.randn(8, 2, device=device)
    loss = torch.nn.functional.mse_loss(model(features), target)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def components(device: str = "cpu"):
    model = torch.nn.Linear(4, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return model, optimizer, scheduler


def test_checkpoint_restores_exact_next_update_and_rng(tmp_path: Path) -> None:
    torch.manual_seed(17)
    np.random.seed(17)
    model, optimizer, scheduler = components()
    update(model, optimizer, scheduler)
    metadata = CheckpointMetadata(
        config_hash="config-sha", scenario_hash="scenario-sha", counters={"updates": 1}
    )
    path = save_training_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
    )

    baseline_loss = update(model, optimizer, scheduler)
    baseline = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    resumed_model, resumed_optimizer, resumed_scheduler = components()
    loaded = load_training_checkpoint(
        path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        expected_config_hash="config-sha",
        expected_scenario_hash="scenario-sha",
        restore_rng=True,
    )
    resumed_loss = update(resumed_model, resumed_optimizer, resumed_scheduler)

    assert loaded == metadata
    assert resumed_loss == baseline_loss
    for name, tensor in resumed_model.state_dict().items():
        assert torch.equal(tensor, baseline[name])
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_rejects_provenance_mismatch(tmp_path: Path) -> None:
    model, optimizer, scheduler = components()
    path = save_training_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=CheckpointMetadata("config", "scenario", {"steps": 0}),
    )

    with pytest.raises(ValueError, match="config hash"):
        load_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_config_hash="wrong",
            expected_scenario_hash="scenario",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_checkpoint_resumes_next_update_exactly(tmp_path: Path) -> None:
    torch.manual_seed(23)
    model, optimizer, scheduler = components("cuda")
    update(model, optimizer, scheduler)
    path = save_training_checkpoint(
        tmp_path / "cuda.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=CheckpointMetadata("config", "scenario", {"updates": 1}),
    )
    baseline_loss = update(model, optimizer, scheduler)
    baseline = {name: value.detach().clone() for name, value in model.state_dict().items()}
    resumed_model, resumed_optimizer, resumed_scheduler = components("cuda")
    load_training_checkpoint(
        path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        expected_config_hash="config",
        expected_scenario_hash="scenario",
        restore_rng=True,
    )

    resumed_loss = update(resumed_model, resumed_optimizer, resumed_scheduler)

    assert resumed_loss == baseline_loss
    assert all(
        torch.equal(value, baseline[name])
        for name, value in resumed_model.state_dict().items()
    )
