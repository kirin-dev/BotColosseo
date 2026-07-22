import pytest
import torch

from botcolosseo.training.gae import (
    generalized_advantage_estimate,
    normalize_advantages,
)


def test_terminal_does_not_bootstrap_but_timeout_does() -> None:
    rewards = torch.tensor([[1.0], [1.0]])
    values = torch.tensor([[0.5], [0.5]])
    next_values = torch.tensor([[2.0], [2.0]])
    terminated = torch.tensor([[True], [False]])
    truncated = torch.tensor([[False], [True]])

    output = generalized_advantage_estimate(
        rewards,
        values,
        next_values,
        terminated,
        truncated,
        gamma=0.9,
        gae_lambda=0.8,
    )

    torch.testing.assert_close(output.advantages[:, 0], torch.tensor([0.5, 2.3]))
    torch.testing.assert_close(output.returns[:, 0], torch.tensor([1.0, 2.8]))


def test_gae_matches_hand_computation_and_stops_at_timeout() -> None:
    output = generalized_advantage_estimate(
        rewards=torch.tensor([[1.0, 2.0, 3.0]]),
        values=torch.tensor([[0.5, 1.0, 1.5]]),
        next_values=torch.tensor([[1.0, 1.5, 4.0]]),
        terminated=torch.tensor([[False, False, False]]),
        truncated=torch.tensor([[False, True, False]]),
        gamma=0.9,
        gae_lambda=0.8,
    )

    torch.testing.assert_close(
        output.advantages, torch.tensor([[3.092, 2.35, 5.1]])
    )
    torch.testing.assert_close(output.returns, torch.tensor([[3.592, 3.35, 6.6]]))


def test_advantage_normalization_uses_valid_entries_only() -> None:
    advantages = torch.tensor([[1.0, 2.0, 100.0]])
    valid = torch.tensor([[True, True, False]])

    normalized = normalize_advantages(advantages, valid)

    torch.testing.assert_close(normalized, torch.tensor([[-1.0, 1.0, 0.0]]))


@pytest.mark.parametrize("name", ["rewards", "values", "next_values"])
def test_gae_rejects_nonfinite_inputs(name: str) -> None:
    tensors = {
        "rewards": torch.ones(1, 2),
        "values": torch.ones(1, 2),
        "next_values": torch.ones(1, 2),
    }
    tensors[name][0, 0] = float("nan")

    with pytest.raises(FloatingPointError):
        generalized_advantage_estimate(
            **tensors,
            terminated=torch.zeros(1, 2, dtype=torch.bool),
            truncated=torch.zeros(1, 2, dtype=torch.bool),
            gamma=0.99,
            gae_lambda=0.95,
        )
