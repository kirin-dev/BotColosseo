from pathlib import Path

import torch

from botcolosseo.agents.model import AsymmetricActorCritic
from botcolosseo.agents.style_model import StyledActorCritic
from botcolosseo.training.style_distillation import (
    StyleDistillationTrainer,
    evaluate_defensive_distillation,
    evaluate_explorer_distillation,
    evaluate_style_distillation,
    load_style_distillation_checkpoint,
    save_style_distillation_checkpoint,
    save_style_neutral_checkpoint,
    style_distillation_loss,
)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "frames": torch.zeros(2, 3, 1, 84, 84, dtype=torch.uint8),
        "scalars": torch.zeros(2, 3, 6),
        "previous_actions": torch.zeros(2, 3, dtype=torch.long),
        "actions": torch.tensor([[9, 1, 0], [10, 5, 0]]),
        "masks": torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 0.0]]),
        "valid": torch.tensor([[True, True, False], [True, True, False]]),
        "present": torch.tensor([[True, True, True], [True, True, False]]),
        "task_ids": torch.tensor([[1, 0, 0], [1, 0, 0]]),
    }


def test_style_distillation_loss_combines_masked_imitation_and_base_kl() -> None:
    base = torch.zeros(1, 2, 13)
    style = base.clone()
    style[0, 0, 9] = 2.0
    actions = torch.tensor([[9, 0]])
    supervised = torch.tensor([[True, False]])
    present = torch.tensor([[True, True]])

    metrics = style_distillation_loss(style, base, actions, supervised, present, beta_kl=0.5)

    assert metrics.supervised_count == 1
    assert metrics.accuracy == 1.0
    assert metrics.base_kl > 0
    torch.testing.assert_close(metrics.total, metrics.imitation + 0.5 * metrics.base_kl)


def test_trainer_updates_only_adapter_and_style_policy() -> None:
    torch.manual_seed(3)
    model = StyledActorCritic.from_base(AsymmetricActorCritic(), bottleneck=8)
    frozen = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.startswith("base.")
    }
    before_adapter = {
        name: value.detach().clone() for name, value in model.adapter.state_dict().items()
    }
    trainer = StyleDistillationTrainer.create(
        model,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip=0.5,
        beta_kl=0.05,
        total_updates=2,
    )

    metrics = trainer.train_step(_batch())

    assert metrics.update == 1
    assert all(torch.equal(model.state_dict()[name], value) for name, value in frozen.items())
    assert any(
        not torch.equal(model.adapter.state_dict()[name], value)
        for name, value in before_adapter.items()
    )


def test_checkpoint_warm_start_is_bound_to_unchanged_m3_base(tmp_path: Path) -> None:
    torch.manual_seed(5)
    model = StyledActorCritic.from_base(AsymmetricActorCritic(), bottleneck=8)
    model.policy.bias.data[9] += 1.0
    path = save_style_distillation_checkpoint(
        tmp_path / "style.pt",
        model=model,
        base_checkpoint_sha256="a" * 64,
        scenario_hash="b" * 64,
        data_manifest_sha256="c" * 64,
        config_hash="d" * 64,
        updates=10,
    )
    target = StyledActorCritic.from_base(model.base, bottleneck=8)

    metadata = load_style_distillation_checkpoint(
        path,
        model=target,
        expected_base_checkpoint_sha256="a" * 64,
        expected_scenario_hash="b" * 64,
    )

    assert metadata["updates"] == 10
    torch.testing.assert_close(target.policy.bias, model.policy.bias)


def test_offline_evaluation_reports_positive_shift_and_negative_collapse() -> None:
    model = StyledActorCritic.from_base(AsymmetricActorCritic(), bottleneck=8)
    model.policy.bias.data[9] += 2.0

    metrics = evaluate_style_distillation(model, (_batch(),))

    assert metrics["positive_count"] == 2
    assert metrics["negative_count"] == 2
    assert metrics["attack_probability_delta"] > 0
    assert metrics["passed"] is (metrics["negative_attack_prediction_rate"] < 0.5)


def test_defensive_evaluation_separates_risk_shift_and_no_risk_drift() -> None:
    model = StyledActorCritic.from_base(AsymmetricActorCritic(), bottleneck=8)
    batch = _batch()
    risk_targets = batch["actions"][batch["task_ids"] != 0]
    target = int(risk_targets[0])
    batch["actions"][1, 0] = target
    model.policy.bias.data[target] += 20.0

    metrics = evaluate_defensive_distillation(model, (batch,))

    assert metrics["risk_count"] == 2
    assert metrics["no_risk_count"] == 2
    assert metrics["risk_target_agreement"] == 1.0
    assert metrics["risk_target_agreement_delta"] > 0


def test_defensive_checkpoint_identity_is_style_bound(tmp_path: Path) -> None:
    model = StyledActorCritic.from_base(AsymmetricActorCritic(), bottleneck=8)
    path = save_style_distillation_checkpoint(
        tmp_path / "defensive.pt",
        model=model,
        base_checkpoint_sha256="a" * 64,
        scenario_hash="b" * 64,
        data_manifest_sha256="c" * 64,
        config_hash="d" * 64,
        updates=10,
        style="defensive",
    )
    target = StyledActorCritic.from_base(model.base, bottleneck=8)

    metadata = load_style_distillation_checkpoint(
        path,
        model=target,
        expected_base_checkpoint_sha256="a" * 64,
        expected_scenario_hash="b" * 64,
        expected_style="defensive",
    )

    assert metadata["updates"] == 10


def test_explorer_evaluation_separates_route_targets_from_base_context() -> None:
    model = StyledActorCritic.from_base(AsymmetricActorCritic(), bottleneck=8)
    batch = _batch()
    with torch.no_grad():
        base = model.base.actor(
            batch["frames"],
            batch["scalars"],
            batch["previous_actions"],
            batch["masks"],
        )
    base_actions = base.logits.argmax(dim=-1)
    target = (int(base_actions[0, 0]) + 1) % 13
    batch["actions"][0, 0] = target
    batch["actions"][1, 0] = target
    batch["actions"][0, 1] = base_actions[0, 1]
    batch["actions"][1, 1] = base_actions[1, 1]
    model.policy.bias.data[target] += 20.0

    metrics = evaluate_explorer_distillation(model, (batch,))

    assert metrics["route_count"] == 2
    assert metrics["base_context_count"] == 2
    assert metrics["route_target_agreement"] == 1.0
    assert metrics["route_target_agreement_delta"] == 1.0


def test_neutral_checkpoint_preserves_zero_update_style_identity(tmp_path: Path) -> None:
    model = StyledActorCritic.from_base(AsymmetricActorCritic(), bottleneck=8)

    path = save_style_neutral_checkpoint(
        tmp_path / "neutral.pt",
        model=model,
        style="defensive",
        base_checkpoint_sha256="a" * 64,
        scenario_hash="b" * 64,
        data_manifest_sha256="c" * 64,
        config_hash="d" * 64,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["kind"] == "style_neutral"
    assert payload["style"] == "defensive"
    assert payload["updates"] == 0
    assert all(
        torch.equal(payload["model"][name], value) for name, value in model.state_dict().items()
    )
