import pytest
import torch

from botcolosseo.training.style_ppo import (
    categorical_style_kl,
    masked_teacher_cross_entropy,
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
    with pytest.raises(ValueError):
        categorical_style_kl(
            torch.zeros(1, 2, 2), torch.zeros(1, 2, 2), torch.zeros(1, 2, dtype=torch.bool)
        )
