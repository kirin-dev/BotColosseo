import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from botcolosseo.agents.league_opponents import (
    CheckpointOpponentPolicy,
    OpponentSpec,
    sha256_file,
)
from botcolosseo.agents.model import AsymmetricActorCritic
from botcolosseo.agents.style_model import StyledActorCritic
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_types import DuelActorObservation


def _observation(previous_action: int = 0) -> DuelActorObservation:
    return DuelActorObservation(
        frame=np.zeros((84, 84), dtype=np.uint8),
        health=100.0,
        armor=0.0,
        ammo=20.0,
        own_score=0,
        opponent_score=0,
        has_core=False,
        previous_action=previous_action,
    )


def _write_checkpoint(path: Path, *, scenario_hash: str = "scenario") -> Path:
    model = AsymmetricActorCritic()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.actor.recurrent.bias_ih[512:] = 1.0
        model.actor.policy.bias[4] = 5.0
    torch.save(
        {
            "schema_version": 1,
            "model": model.state_dict(),
            "metadata": {
                "config_hash": "config",
                "scenario_hash": scenario_hash,
                "counters": {"environment_steps": 1},
            },
        },
        path,
    )
    return path


def _spec(path: Path, **changes: object) -> OpponentSpec:
    values: dict[str, object] = {
        "opponent_id": "history-0001",
        "kind": "checkpoint",
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "scenario_hash": "scenario",
        "selection_evidence": "reports/m3/admission-0001.json",
    }
    values.update(changes)
    return OpponentSpec(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    [
        {"opponent_id": ""},
        {"kind": "unknown"},
        {"checkpoint": None},
        {"checkpoint_sha256": "bad"},
        {"scenario_hash": ""},
        {"selection_evidence": ""},
    ],
)
def test_checkpoint_opponent_spec_rejects_invalid_fields(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    path = _write_checkpoint(tmp_path / "policy.pt")

    with pytest.raises(ValueError):
        _spec(path, **changes)


def test_script_opponent_spec_forbids_checkpoint_fields() -> None:
    with pytest.raises(ValueError, match="Script"):
        OpponentSpec(
            opponent_id="objective_first",
            kind="script",
            checkpoint="policy.pt",
            checkpoint_sha256="a" * 64,
            scenario_hash="scenario",
            selection_evidence="builtin:objective_first",
        )


def test_checkpoint_policy_rejects_hash_and_scenario_mismatch(tmp_path: Path) -> None:
    path = _write_checkpoint(tmp_path / "policy.pt")

    with pytest.raises(ValueError, match="hash"):
        CheckpointOpponentPolicy.load(
            _spec(path, checkpoint_sha256="0" * 64), device=torch.device("cpu")
        )
    with pytest.raises(ValueError, match="scenario"):
        CheckpointOpponentPolicy.load(
            _spec(path, scenario_hash="different"), device=torch.device("cpu")
        )


def test_checkpoint_policy_is_greedy_recurrent_and_reset_only_at_episode_boundary(
    tmp_path: Path,
) -> None:
    path = _write_checkpoint(tmp_path / "policy.pt")
    policy = CheckpointOpponentPolicy.load(_spec(path), device=torch.device("cpu"))

    assert all(not parameter.requires_grad for parameter in policy._actor.parameters())
    assert list(inspect.signature(CheckpointOpponentPolicy.act).parameters) == [
        "self",
        "observation",
    ]
    with pytest.raises(RuntimeError, match="reset"):
        policy.act(_observation())

    policy.reset()
    assert policy.act(_observation()).value == 4
    first_hidden = policy._hidden.detach().clone()
    assert policy.act(_observation(previous_action=4)).value == 4
    assert not torch.equal(policy._hidden, torch.zeros_like(policy._hidden))

    policy.reset()
    assert torch.equal(policy._hidden, torch.zeros_like(policy._hidden))
    assert policy.act(_observation()).value == 4
    assert torch.equal(policy._hidden, first_hidden)


def test_checkpoint_policy_fork_shares_frozen_actor_but_not_recurrent_state(
    tmp_path: Path,
) -> None:
    path = _write_checkpoint(tmp_path / "policy.pt")
    first = CheckpointOpponentPolicy.load(_spec(path), device=torch.device("cpu"))
    second = first.fork()

    first.reset()
    second.reset()
    first.act(_observation())

    assert first._actor is second._actor
    assert not torch.equal(first._hidden, second._hidden)
    assert all(not parameter.requires_grad for parameter in second._actor.parameters())


def test_checkpoint_policy_loads_auditable_league_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "candidate.pt"
    model = AsymmetricActorCritic()
    torch.save(
        {
            "schema_version": 1,
            "identity": {
                "base_checkpoint_sha256": "a" * 64,
                "config_hash": "b" * 64,
                "train_manifest_hash": "c" * 64,
                "pool_manifest_hash": "d" * 64,
                "payoff_report_hash": "e" * 64,
                "scenario_hash": "scenario",
            },
            "state": {
                "environment_steps": 200_000,
                "updates": 10,
                "episodes": 4,
                "next_pair_slot": 2,
            },
            "model": model.state_dict(),
        },
        path,
    )

    policy = CheckpointOpponentPolicy.load(_spec(path), device=torch.device("cpu"))

    policy.reset()
    assert isinstance(policy.act(_observation()), MacroAction)


def test_checkpoint_policy_loads_style_adapter_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "aggressive.pt"
    model = StyledActorCritic.from_base(AsymmetricActorCritic(), bottleneck=16)
    torch.save(
        {
            "schema_version": 1,
            "identity": {
                "base_checkpoint_sha256": "a" * 64,
                "config_hash": "b" * 64,
                "train_manifest_hash": "c" * 64,
                "pool_manifest_hash": "d" * 64,
                "payoff_report_hash": "e" * 64,
                "scenario_hash": "scenario",
            },
            "state": {
                "environment_steps": 1,
                "updates": 1,
                "episodes": 0,
                "next_pair_slot": 0,
            },
            "model": model.state_dict(),
        },
        path,
    )

    policy = CheckpointOpponentPolicy.load(_spec(path), device=torch.device("cpu"))
    policy.reset()

    assert isinstance(policy.act(_observation()), MacroAction)


@pytest.mark.parametrize("style", ("aggressive", "defensive"))
def test_checkpoint_policy_loads_style_distillation_checkpoint(
    tmp_path: Path, style: str
) -> None:
    path = tmp_path / f"{style}-distilled.pt"
    model = StyledActorCritic.from_base(AsymmetricActorCritic(), bottleneck=16)
    torch.save(
        {
            "schema_version": 1,
            "kind": "style_distillation",
            "style": style,
            "base_checkpoint_sha256": "a" * 64,
            "scenario_hash": "scenario",
            "data_manifest_sha256": "b" * 64,
            "config_hash": "c" * 64,
            "updates": 10,
            "model": model.state_dict(),
        },
        path,
    )

    policy = CheckpointOpponentPolicy.load(_spec(path), device=torch.device("cpu"))
    policy.reset()

    assert isinstance(policy.act(_observation()), MacroAction)


def test_checkpoint_policy_loads_style_interpolation_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "aggressive-alpha.pt"
    model = StyledActorCritic.from_base(AsymmetricActorCritic(), bottleneck=16)
    torch.save(
        {
            "schema_version": 1,
            "kind": "style_interpolation",
            "style": "aggressive",
            "alpha": 0.5,
            "base_checkpoint_sha256": "a" * 64,
            "scenario_hash": "scenario",
            "distilled_checkpoint_sha256": "b" * 64,
            "ppo_checkpoint_sha256": "c" * 64,
            "interpolation_sha256": "d" * 64,
            "model": model.state_dict(),
        },
        path,
    )

    policy = CheckpointOpponentPolicy.load(_spec(path), device=torch.device("cpu"))
    policy.reset()

    assert isinstance(policy.act(_observation()), MacroAction)


@pytest.mark.parametrize("style", ("defensive", "explorer"))
def test_checkpoint_policy_loads_neutral_style_interpolation_checkpoint(
    tmp_path: Path,
    style: str,
) -> None:
    path = tmp_path / f"{style}-alpha.pt"
    model = StyledActorCritic.from_base(AsymmetricActorCritic(), bottleneck=16)
    torch.save(
        {
            "schema_version": 1,
            "kind": "style_interpolation",
            "style": style,
            "alpha": 0.5,
            "base_checkpoint_sha256": "a" * 64,
            "scenario_hash": "scenario",
            "distilled_checkpoint_sha256": "b" * 64,
            "neutral_checkpoint_sha256": "c" * 64,
            "interpolation_sha256": "d" * 64,
            "model": model.state_dict(),
        },
        path,
    )

    policy = CheckpointOpponentPolicy.load(_spec(path), device=torch.device("cpu"))
    policy.reset()

    assert isinstance(policy.act(_observation()), MacroAction)
