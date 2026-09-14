import torch

from botcolosseo.agents.hierarchical_model import (
    CommandActorCritic,
    CommandExecutor,
    StrategicActor,
)
from botcolosseo.agents.model import RecurrentActor
from botcolosseo.data.extraction_demonstrations import EXTRACTION_SCALAR_DIM


def test_zero_init_preserves_base_and_chunked_switch_keeps_memory():
    torch.manual_seed(3)
    torch.set_num_threads(1)
    base = RecurrentActor(scalar_dim=EXTRACTION_SCALAR_DIM)
    actor = CommandExecutor()
    actor.initialize_from(base)
    frames = torch.randint(256, (1, 3, 1, 84, 84), dtype=torch.uint8)
    scalars = torch.zeros(1, 3, EXTRACTION_SCALAR_DIM)
    actions = torch.zeros(1, 3, dtype=torch.long)
    masks = torch.ones(1, 3)
    commands = torch.tensor([[0, 3, 6]])
    difficulty = torch.tensor([[[1.0], [0.5], [0.0]]])
    result = actor(frames, scalars, actions, masks, commands, difficulty)
    torch.testing.assert_close(result.logits, base(frames, scalars, actions, masks).logits)
    state = None
    chunks = []
    for t in range(3):
        out = actor(
            frames[:, t : t + 1],
            scalars[:, t : t + 1],
            actions[:, t : t + 1],
            masks[:, t : t + 1],
            commands[:, t : t + 1],
            difficulty[:, t : t + 1],
            state,
        )
        chunks.append(out.logits)
        state = out.hidden
    torch.testing.assert_close(torch.cat(chunks, 1), result.logits)
    result.logits.square().mean().backward()
    assert actor.command_output.network[-1].weight.grad.abs().sum() > 0
    assert actor.difficulty_output.network[-1].weight.grad.abs().sum() > 0


def test_privileged_critic_does_not_change_actor_or_backpropagate_into_it():
    model = CommandActorCritic(CommandExecutor())
    inputs = (
        torch.zeros(1, 1, 1, 84, 84),
        torch.zeros(1, 1, EXTRACTION_SCALAR_DIM),
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1),
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, 1),
    )
    first = model(*inputs, torch.zeros(1, 1, 20))
    second = model(*inputs, torch.ones(1, 1, 20))
    torch.testing.assert_close(first.logits, second.logits)
    second.values.sum().backward()
    assert all(p.grad is None for p in model.actor.parameters())
    assert model.value[-1].weight.grad is not None


def test_strategic_gradient_cannot_enter_executor_features():
    actor = StrategicActor()
    features = torch.randn(2, 3, 256, requires_grad=True)
    out = actor(
        features,
        torch.zeros(2, 3, EXTRACTION_SCALAR_DIM),
        torch.zeros(2, 3, dtype=torch.long),
        torch.ones(2, 3, 1),
        torch.ones(2, 3, 3),
        torch.ones(2, 3, 1),
        torch.ones(2, 3),
    )
    assert out.logits.shape == (2, 3, 7)
    out.logits.square().mean().backward()
    assert features.grad is None
    assert actor.style_output.network[-1].weight.grad.abs().sum() > 0
    assert actor.difficulty_output.network[-1].weight.grad.abs().sum() > 0
