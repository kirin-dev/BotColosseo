from pathlib import Path

import torch

from botcolosseo.agents.model import AsymmetricActorCritic
from botcolosseo.agents.style_model import StyledActorCritic
from botcolosseo.training.style_distillation import (
    StyleDistillationTrainer,
    evaluate_style_distillation,
    load_style_distillation_checkpoint,
    save_style_distillation_checkpoint,
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
