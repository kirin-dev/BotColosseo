from pathlib import Path

import pytest
import torch

from botcolosseo.agents.model import AsymmetricActorCritic
from botcolosseo.agents.style_model import StyledActorCritic
from botcolosseo.training.style_interpolation import (
    interpolate_defensive_checkpoints,
    interpolate_style_checkpoints,
)


def _sources(tmp_path: Path) -> tuple[Path, Path, StyledActorCritic, StyledActorCritic]:
    torch.manual_seed(7)
    base = AsymmetricActorCritic()
    distilled = StyledActorCritic.from_base(base, bottleneck=8)
    ppo = StyledActorCritic.from_base(base, bottleneck=8)
    with torch.no_grad():
        for parameter in distilled.adapter.parameters():
            parameter.fill_(2.0)
        for parameter in distilled.policy.parameters():
            parameter.fill_(2.0)
        for parameter in ppo.adapter.parameters():
            parameter.fill_(0.0)
        for parameter in ppo.policy.parameters():
            parameter.fill_(0.0)
    distilled_path = tmp_path / "distilled.pt"
    ppo_path = tmp_path / "ppo.pt"
    torch.save(
        {
            "schema_version": 1,
            "kind": "style_distillation",
            "style": "aggressive",
            "base_checkpoint_sha256": "a" * 64,
            "scenario_hash": "scenario",
            "model": distilled.state_dict(),
        },
        distilled_path,
    )
    torch.save(
        {
            "schema_version": 1,
            "identity": {
                "base_checkpoint_sha256": "a" * 64,
                "scenario_hash": "scenario",
            },
            "model": ppo.state_dict(),
        },
        ppo_path,
    )
    return distilled_path, ppo_path, distilled, ppo


def test_interpolation_changes_only_style_branch_and_binds_sources(tmp_path: Path) -> None:
    distilled_path, ppo_path, distilled, _ = _sources(tmp_path)
    output = tmp_path / "alpha-050.pt"

    report = interpolate_style_checkpoints(distilled_path, ppo_path, output, alpha=0.5)
    payload = torch.load(output, map_location="cpu", weights_only=False)

    assert report["frozen_base_actor"] is True
    assert payload["kind"] == "style_interpolation"
    assert payload["alpha"] == 0.5
    for name, value in payload["model"].items():
        if name.startswith(("adapter.", "policy.")):
            torch.testing.assert_close(value, torch.ones_like(value))
        elif name.startswith("base.actor."):
            torch.testing.assert_close(value, distilled.state_dict()[name])


@pytest.mark.parametrize("alpha", (0.0, 1.0, -0.1, 1.1))
def test_interpolation_rejects_endpoint_or_out_of_range_alpha(tmp_path: Path, alpha: float) -> None:
    distilled_path, ppo_path, _, _ = _sources(tmp_path)

    with pytest.raises(ValueError, match="alpha"):
        interpolate_style_checkpoints(distilled_path, ppo_path, tmp_path / "out.pt", alpha=alpha)


def test_interpolation_rejects_changed_frozen_actor(tmp_path: Path) -> None:
    distilled_path, ppo_path, _, _ = _sources(tmp_path)
    payload = torch.load(ppo_path, map_location="cpu", weights_only=False)
    payload["model"]["base.actor.policy.bias"] += 1.0
    torch.save(payload, ppo_path)

    with pytest.raises(ValueError, match="frozen Strong Base"):
        interpolate_style_checkpoints(distilled_path, ppo_path, tmp_path / "out.pt", alpha=0.5)


def test_defensive_interpolation_uses_neutral_branch_and_preserves_base(
    tmp_path: Path,
) -> None:
    base = AsymmetricActorCritic()
    neutral = StyledActorCritic.from_base(base, bottleneck=8)
    distilled = StyledActorCritic.from_base(base, bottleneck=8)
    with torch.no_grad():
        for parameter in distilled.adapter.parameters():
            parameter.add_(2.0)
        for parameter in distilled.policy.parameters():
            parameter.add_(2.0)
    identity = {
        "base_checkpoint_sha256": "a" * 64,
        "scenario_hash": "scenario",
        "data_manifest_sha256": "b" * 64,
        "config_hash": "c" * 64,
    }
    neutral_path = tmp_path / "neutral.pt"
    distilled_path = tmp_path / "distilled.pt"
    torch.save(
        {
            "schema_version": 1,
            "kind": "style_neutral",
            "style": "defensive",
            "updates": 0,
            "model": neutral.state_dict(),
            **identity,
        },
        neutral_path,
    )
    torch.save(
        {
            "schema_version": 1,
            "kind": "style_distillation",
            "style": "defensive",
            "updates": 10,
            "model": distilled.state_dict(),
            **identity,
        },
        distilled_path,
    )
    output = tmp_path / "alpha-050.pt"

    report = interpolate_defensive_checkpoints(
        neutral_path, distilled_path, output, alpha=0.5
    )
    payload = torch.load(output, map_location="cpu", weights_only=False)

    assert report["style"] == "defensive"
    assert payload["neutral_checkpoint_sha256"] == report["neutral_checkpoint_sha256"]
    for name, value in payload["model"].items():
        if name.startswith(("adapter.", "policy.")):
            expected = torch.lerp(
                neutral.state_dict()[name], distilled.state_dict()[name], 0.5
            )
            torch.testing.assert_close(value, expected)
        elif name.startswith("base."):
            torch.testing.assert_close(value, neutral.state_dict()[name])
