import pytest
import torch

from botcolosseo.training.hierarchical_smdp import aggregate_strategic_transitions, strategic_gae


def test_variable_duration_gae_and_truncation_trace_boundary():
    rewards = torch.tensor([[1.0, 2.0, 100.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([[0.0, 4.0, 0.0]])
    durations = torch.tensor([[2, 3, 1]])
    terminal = torch.tensor([[False, False, True]])
    truncated = torch.tensor([[False, True, False]])
    result = strategic_gae(
        rewards, values, next_values, durations, terminal, truncated, gamma=0.5, gae_lambda=0.8
    )
    # Truncated middle transition bootstraps 4, but never sees next episode's 100.
    torch.testing.assert_close(result.advantages, torch.tensor([[1.5, 2.5, 100.0]]))
    terminal[0, 1] = True
    truncated[0, 1] = False
    result = strategic_gae(
        rewards, values, next_values, durations, terminal, truncated, gamma=0.5, gae_lambda=0.8
    )
    torch.testing.assert_close(result.returns, torch.tensor([[1.4, 2.0, 100.0]]))


@pytest.mark.parametrize("terminal", [True, False])
def test_actual_duration_condition_and_boundary(terminal):
    rows = [
        dict(
            decision=i,
            replanned=i in (0, 2),
            command=0 if i < 2 else 3,
            command_log_prob=-0.5 if i in (0, 2) else None,
            style=(0, 0, 0) if i < 2 else (1, 0, 0),
            high_difficulty=1,
        )
        for i in range(5)
    ]
    result = aggregate_strategic_transitions(
        rows, [1] * 5, gamma=0.9, terminated=terminal, truncated=not terminal
    )
    assert [r.duration for r in result] == [2, 3]
    assert result[0].reward == pytest.approx(1.9)
    assert result[1].reward == pytest.approx(2.71)
    assert result[0].bootstrap_discount == pytest.approx(0.81)
    assert result[1].bootstrap_discount == pytest.approx(0 if terminal else 0.729)
    assert result[1].condition.aggressive == 1
    rows[1]["command"] = 6
    with pytest.raises(ValueError, match="without replan"):
        aggregate_strategic_transitions(
            rows, [1] * 5, gamma=0.9, terminated=terminal, truncated=not terminal
        )
