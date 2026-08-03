from __future__ import annotations

from pathlib import Path

import pytest
import torch

from botcolosseo.agents.checkpoint import (
    CheckpointMetadata,
    save_training_checkpoint,
)
from botcolosseo.agents.extraction_model import (
    ExtractionResidualStyleActorCritic,
    create_extraction_actor,
    create_extraction_actor_critic,
)
from botcolosseo.envs.actions import MacroAction
from botcolosseo.training.extraction_checkpoint import (
    load_extraction_bc_warm_start,
    load_extraction_strong_actor,
    load_extraction_strong_actor_critic,
    load_extraction_style_actor,
    sha256_file,
    validate_extraction_teacher_lineage,
)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    scenario: str,
    *,
    lineage: dict[str, str | int] | None = None,
) -> Path:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    return save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=CheckpointMetadata(
            "config", scenario, {"updates": 3}, dict(lineage or {})
        ),
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


def test_teacher_lineage_preflight_rejects_mismatch(tmp_path: Path) -> None:
    teacher_sha256 = "a" * 64
    checkpoint = save_checkpoint(
        tmp_path / "aligned-bc.pt",
        create_extraction_actor(),
        "scenario",
        lineage={"teacher_implementation_sha256": teacher_sha256},
    )

    metadata = validate_extraction_teacher_lineage(
        checkpoint,
        expected_scenario_hash="scenario",
        expected_teacher_sha256=teacher_sha256,
        expected_sha256=sha256_file(checkpoint),
    )

    assert metadata.lineage["teacher_implementation_sha256"] == teacher_sha256
    with pytest.raises(ValueError, match="Teacher identities"):
        validate_extraction_teacher_lineage(
            checkpoint,
            expected_scenario_hash="scenario",
            expected_teacher_sha256="b" * 64,
        )


def test_style_loader_proves_frozen_strong_actor_identity(tmp_path: Path) -> None:
    base = create_extraction_actor_critic()
    base_checkpoint = save_checkpoint(tmp_path / "base.pt", base, "scenario")
    style = ExtractionResidualStyleActorCritic(base)
    style_checkpoint = save_checkpoint(tmp_path / "style.pt", style, "scenario")

    actor, _ = load_extraction_style_actor(
        style_checkpoint,
        base_checkpoint=base_checkpoint,
        expected_scenario_hash="scenario",
        expected_base_sha256=sha256_file(base_checkpoint),
        bottleneck=32,
        max_delta=2,
    )

    assert actor.hidden_size == 256
    changed = ExtractionResidualStyleActorCritic(base)
    with torch.no_grad():
        changed.base.actor.policy.bias.add_(1)
    changed_checkpoint = save_checkpoint(
        tmp_path / "changed.pt",
        changed,
        "scenario",
    )
    with pytest.raises(ValueError, match="frozen Strong"):
        load_extraction_style_actor(
            changed_checkpoint,
            base_checkpoint=base_checkpoint,
            expected_scenario_hash="scenario",
            expected_base_sha256=sha256_file(base_checkpoint),
            bottleneck=32,
            max_delta=2,
        )


def test_defensive_style_loader_blocks_attacks_only_at_low_health(
    tmp_path: Path,
) -> None:
    base = create_extraction_actor_critic()
    with torch.no_grad():
        base.actor.policy.weight.zero_()
        base.actor.policy.bias.zero_()
        base.actor.policy.bias[int(MacroAction.ATTACK)] = 10
        base.actor.policy.bias[int(MacroAction.MOVE_FORWARD)] = 1
    base_checkpoint = save_checkpoint(tmp_path / "base.pt", base, "scenario")
    style = ExtractionResidualStyleActorCritic(base)
    style_checkpoint = save_checkpoint(tmp_path / "style.pt", style, "scenario")
    actor, _ = load_extraction_style_actor(
        style_checkpoint,
        base_checkpoint=base_checkpoint,
        expected_scenario_hash="scenario",
        expected_base_sha256=sha256_file(base_checkpoint),
        bottleneck=32,
        max_delta=2,
        defensive_guardrail=True,
    )
    frames = torch.zeros(1, 1, 1, 84, 84, dtype=torch.uint8)
    previous_actions = torch.zeros(1, 1, dtype=torch.long)
    masks = torch.ones(1, 1)
    healthy = torch.zeros(1, 1, 9)
    healthy[..., 0] = 1
    healthy[..., 1] = 1
    low_health = healthy.clone()
    low_health[..., 0] = 0.4
    low_health[..., 2] = 25 / 150
    low_health_without_value = low_health.clone()
    low_health_without_value[..., 2] = 0
    low_ammo_with_value = healthy.clone()
    low_ammo_with_value[..., 1] = 0.125
    low_ammo_with_value[..., 2] = 25 / 150

    healthy_output = actor(frames, healthy, previous_actions, masks)
    guarded_output = actor(frames, low_health, previous_actions, masks)
    unguarded_output = actor(
        frames,
        low_health_without_value,
        previous_actions,
        masks,
    )
    low_ammo_output = actor(
        frames,
        low_ammo_with_value,
        previous_actions,
        masks,
    )

    assert healthy_output.logits.argmax(-1).item() == int(MacroAction.ATTACK)
    assert unguarded_output.logits.argmax(-1).item() == int(MacroAction.ATTACK)
    assert low_ammo_output.logits.argmax(-1).item() == int(MacroAction.ATTACK)
    assert guarded_output.logits.argmax(-1).item() == int(
        MacroAction.MOVE_FORWARD
    )
