import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.agents.model import AsymmetricActorCritic
from botcolosseo.training.league_checkpoint import (
    LeagueCheckpointState,
    LeagueRunIdentity,
    load_league_checkpoint,
    save_league_checkpoint,
    warm_start_from_m2,
)


def _identity() -> LeagueRunIdentity:
    return LeagueRunIdentity(
        base_checkpoint_sha256="1" * 64,
        config_hash="2" * 64,
        train_manifest_hash="3" * 64,
        pool_manifest_hash="4" * 64,
        payoff_report_hash="5" * 64,
        scenario_hash="6" * 64,
    )


def _components():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return model, optimizer, scheduler


def _update(model, optimizer, scheduler) -> float:
    optimizer.zero_grad(set_to_none=True)
    offset = random.random() + float(np.random.random())
    features = torch.randn(4, 3) + offset
    target = torch.randn(4, 2)
    loss = torch.nn.functional.mse_loss(model(features), target)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def test_m2_warm_start_loads_model_only_and_verifies_provenance(tmp_path: Path) -> None:
    source = AsymmetricActorCritic()
    with torch.no_grad():
        source.actor.policy.bias.fill_(3.0)
    path = tmp_path / "m2.pt"
    torch.save(
        {
            "schema_version": 1,
            "model": source.state_dict(),
            "metadata": {
                "config_hash": "m2-config",
                "scenario_hash": "6" * 64,
                "counters": {"environment_steps": 800_000},
            },
        },
        path,
    )
    target = AsymmetricActorCritic()
    optimizer = torch.optim.Adam(target.parameters(), lr=1e-4)

    metadata = warm_start_from_m2(
        path,
        target,
        expected_checkpoint_sha256=sha256_file(path),
        expected_scenario_hash="6" * 64,
    )

    assert metadata.counters["environment_steps"] == 800_000
    assert optimizer.state == {}
    assert all(
        torch.equal(value, source.state_dict()[name])
        for name, value in target.state_dict().items()
    )
    with pytest.raises(ValueError, match="hash"):
        warm_start_from_m2(
            path,
            target,
            expected_checkpoint_sha256="0" * 64,
            expected_scenario_hash="6" * 64,
        )


def test_league_checkpoint_restores_exact_next_update_and_rng(tmp_path: Path) -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    model, optimizer, scheduler = _components()
    _update(model, optimizer, scheduler)
    state = LeagueCheckpointState(
        environment_steps=2_048,
        updates=1,
        episodes=4,
        next_pair_slot=2,
    )
    path = save_league_checkpoint(
        tmp_path / "league.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        identity=_identity(),
        state=state,
    )

    baseline_losses = [_update(model, optimizer, scheduler) for _ in range(10)]
    baseline = {name: value.detach().clone() for name, value in model.state_dict().items()}
    resumed_model, resumed_optimizer, resumed_scheduler = _components()
    loaded = load_league_checkpoint(
        path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        expected_identity=_identity(),
        restore_rng=True,
    )
    resumed_losses = [
        _update(resumed_model, resumed_optimizer, resumed_scheduler) for _ in range(10)
    ]

    assert loaded == state
    assert resumed_losses == baseline_losses
    assert all(
        torch.equal(value, baseline[name])
        for name, value in resumed_model.state_dict().items()
    )
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "field",
    [
        "base_checkpoint_sha256",
        "config_hash",
        "train_manifest_hash",
        "pool_manifest_hash",
        "payoff_report_hash",
        "scenario_hash",
    ],
)
def test_league_resume_rejects_each_identity_drift(tmp_path: Path, field: str) -> None:
    model, optimizer, scheduler = _components()
    path = save_league_checkpoint(
        tmp_path / "league.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        identity=_identity(),
        state=LeagueCheckpointState(100, 1, 2, 1),
    )
    changed = replace(_identity(), **{field: "a" * 64})

    with pytest.raises(ValueError, match="identity"):
        load_league_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_identity=changed,
        )


def test_league_checkpoint_binds_pair_slot_to_episode_index(tmp_path: Path) -> None:
    model, optimizer, scheduler = _components()

    save_league_checkpoint(
        tmp_path / "odd-side.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        identity=_identity(),
        state=LeagueCheckpointState(100, 1, 3, 1),
    )
    with pytest.raises(ValueError, match="pair slot"):
        save_league_checkpoint(
            tmp_path / "league.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            identity=_identity(),
            state=LeagueCheckpointState(100, 1, 3, 2),
        )
