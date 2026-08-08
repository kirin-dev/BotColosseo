import pytest
import torch

from botcolosseo.training.style_ppo import (
    StylePPOTrainer,
    categorical_style_kl,
    masked_teacher_cross_entropy,
    partitioned_style_kl,
    preferred_action_set_margin_loss,
)


def test_style_kl_is_zero_for_identical_policy_and_positive_after_drift() -> None:
    base = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    valid = torch.tensor([[True, False]])

    same = categorical_style_kl(base, base, valid)
    drifted = categorical_style_kl(torch.tensor([[[3.0, -2.0], [0.0, 1.0]]]), base, valid)

    assert same.item() == pytest.approx(0.0, abs=1e-7)
    assert drifted.item() > 0


def test_style_kl_rejects_invalid_shapes_and_empty_mask() -> None:
    with pytest.raises(ValueError):
        categorical_style_kl(
            torch.zeros(1, 2, 3), torch.zeros(1, 2, 2), torch.ones(1, 2, dtype=torch.bool)
        )


def test_masked_teacher_loss_ignores_unsupervised_tokens() -> None:
    logits = torch.tensor([[[3.0, -1.0], [-5.0, 5.0], [-2.0, 2.0]]])
    actions = torch.tensor([[0, 0, 1]])
    supervised = torch.tensor([[True, False, True]])

    loss, agreement, count = masked_teacher_cross_entropy(
        logits, actions, supervised
    )
    changed = logits.clone()
    changed[0, 1] = torch.tensor([100.0, -100.0])
    changed_loss, _, _ = masked_teacher_cross_entropy(
        changed, actions, supervised
    )

    torch.testing.assert_close(loss, changed_loss)
    assert agreement == 1.0
    assert count == 2


def test_masked_teacher_loss_rejects_empty_mask() -> None:
    with pytest.raises(ValueError, match="at least one"):
        masked_teacher_cross_entropy(
            torch.zeros(1, 2, 13),
            torch.zeros(1, 2, dtype=torch.long),
            torch.zeros(1, 2, dtype=torch.bool),
        )


def test_partitioned_style_kl_separates_opportunity_tokens() -> None:
    base = torch.zeros(1, 3, 2)
    style = base.clone()
    style[0, 0] = torch.tensor([4.0, -4.0])
    style[0, 1] = torch.tensor([-3.0, 3.0])
    valid = torch.tensor([[True, True, False]])
    opportunity = torch.tensor([[True, False, True]])

    inside, outside, inside_count, outside_count = partitioned_style_kl(
        style, base, valid, opportunity
    )

    assert inside.item() > 0
    assert outside.item() > 0
    assert inside_count == 1
    assert outside_count == 1


def test_preferred_action_margin_rewards_probability_lift_over_base() -> None:
    base = torch.zeros(1, 2, 3)
    style = base.clone()
    style[0, 0, 1] = 2.0
    preferred = torch.zeros_like(style, dtype=torch.bool)
    preferred[0, 0, 1:] = True
    supervised = torch.tensor([[True, False]])

    loss, lift, count = preferred_action_set_margin_loss(
        style,
        base,
        preferred,
        supervised,
        margin=0.5,
    )

    assert lift > 0
    assert loss.item() < 0.5
    assert count == 1


def test_preferred_action_margin_rejects_empty_preference_set() -> None:
    with pytest.raises(ValueError, match="preferred action"):
        preferred_action_set_margin_loss(
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2, dtype=torch.bool),
            torch.ones(1, 1, dtype=torch.bool),
            margin=0.1,
        )
    with pytest.raises(ValueError):
        categorical_style_kl(
            torch.zeros(1, 2, 2), torch.zeros(1, 2, 2), torch.zeros(1, 2, dtype=torch.bool)
        )


def test_style_trainer_rejects_negative_residual_penalty() -> None:
    with pytest.raises(ValueError, match="coefficients"):
        StylePPOTrainer(
            torch.nn.Linear(1, 1),
            torch.optim.Adam([torch.nn.Parameter(torch.ones(1))]),
            torch.optim.lr_scheduler.StepLR(
                torch.optim.Adam([torch.nn.Parameter(torch.ones(1))]),
                step_size=1,
            ),
            beta_kl=0,
            rho_residual=-1,
            gradient_clip=1,
            policy_clip=0.2,
            value_clip=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.01,
            max_kl=0.1,
        )
