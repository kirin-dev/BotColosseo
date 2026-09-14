import pytest
import torch

from botcolosseo.agents.hierarchical_model import CommandExecutor
from botcolosseo.data.extraction_demonstrations import EXTRACTION_SCALAR_DIM
from botcolosseo.training.hierarchical_bc import (
    command_weights,
    evaluate_episode,
    fit_counterfactual_episode,
    fit_episode,
)


def test_missing_command_rejected():
    with pytest.raises(ValueError):
        command_weights(torch.tensor([1, 1, 1, 1, 0, 1, 1]))


def test_endpoint_counterfactual_ignores_other_command_labels():
    torch.set_num_threads(1)
    actor = CommandExecutor()
    batch = {
        "frames": torch.zeros(1, 2, 1, 84, 84, dtype=torch.uint8),
        "scalars": torch.zeros(1, 2, EXTRACTION_SCALAR_DIM),
        "previous_actions": torch.zeros(1, 2, dtype=torch.long),
        "commands": torch.tensor([[0, 5]]),
        "difficulty": torch.ones(1, 2, 1),
        "masks": torch.tensor([[0.0, 1.0]]),
    }
    labels = torch.zeros(1, 2, 7, dtype=torch.long)
    valid = torch.zeros_like(labels, dtype=torch.bool)
    valid[..., 5:] = True
    optimizer = torch.optim.SGD(actor.parameters(), lr=0)
    first = fit_counterfactual_episode(actor, batch, labels, valid, optimizer, stride=1)
    labels[..., :5] = 12
    second = fit_counterfactual_episode(actor, batch, labels, valid, optimizer, stride=1)
    assert second == pytest.approx(first)
    labels[..., 5:] = 12
    third = fit_counterfactual_episode(actor, batch, labels, valid, optimizer, stride=1)
    assert third != pytest.approx(first)


def test_tbptt_updates_conditioning_with_masked_labels():
    torch.set_num_threads(1)
    actor = CommandExecutor()
    batch = {
        "frames": torch.zeros(1, 4, 1, 84, 84, dtype=torch.uint8),
        "scalars": torch.zeros(1, 4, EXTRACTION_SCALAR_DIM),
        "previous_actions": torch.zeros(1, 4, dtype=torch.long),
        "actions": torch.tensor([[0, 1, 2, 3]]),
        "commands": torch.tensor([[0, 3, 4, 6]]),
        "difficulty": torch.ones(1, 4, 1),
        "masks": torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
        "valid": torch.tensor([[True, True, False, True]]),
    }
    result = fit_episode(
        actor,
        batch,
        torch.optim.Adam(actor.parameters(), lr=1e-3),
        command_weights(torch.ones(7)),
        chunk_size=2,
    )
    assert result["loss"] > 0
    assert actor.command_output.network[-1].weight.abs().sum() > 0
    before = actor.command_output.network[-1].weight.detach().clone()
    loss = fit_counterfactual_episode(
        actor,
        batch,
        torch.arange(7)[None, None].expand(1, 4, 7),
        torch.ones(1, 4, 7, dtype=torch.bool),
        torch.optim.Adam(actor.parameters(), lr=1e-3),
        stride=1,
    )
    assert loss > 0
    assert not torch.equal(before, actor.command_output.network[-1].weight)
    stats = evaluate_episode(actor, batch, chunk_size=2)
    # Exclude initial command and invalid switch; include switches across chunks.
    assert stats["switch_counts"].tolist() == [0, 0, 0, 1, 0, 0, 1]
    optimizer = torch.optim.SGD(actor.parameters(), lr=0)
    first = fit_episode(actor, batch, optimizer, torch.ones(7), normalization_mass=2)
    second = fit_episode(actor, batch, optimizer, torch.ones(7), normalization_mass=4)
    assert first["loss"] == pytest.approx(2 * second["loss"])
