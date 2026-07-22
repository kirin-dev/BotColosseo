import pytest
import torch

from botcolosseo.training.style_ppo import categorical_style_kl


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
    with pytest.raises(ValueError):
        categorical_style_kl(
            torch.zeros(1, 2, 2), torch.zeros(1, 2, 2), torch.zeros(1, 2, dtype=torch.bool)
        )
