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
