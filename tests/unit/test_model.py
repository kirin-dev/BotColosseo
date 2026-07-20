import pytest
import torch

from botcolosseo.agents.model import AsymmetricActorCritic, RecurrentActor


def inputs(batch: int = 2, time: int = 3, *, device: str = "cpu"):
    generator = torch.Generator(device=device).manual_seed(7)
    frames = torch.rand(batch, time, 1, 84, 84, generator=generator, device=device)
    scalars = torch.rand(batch, time, 6, generator=generator, device=device)
    actions = torch.randint(0, 13, (batch, time), generator=generator, device=device)
    masks = torch.ones(batch, time, device=device)
    privileged = torch.rand(batch, time, 12, generator=generator, device=device)
    return frames, scalars, actions, masks, privileged


def test_actor_and_asymmetric_critic_shapes_backward_and_fairness() -> None:
    model = AsymmetricActorCritic()
    frames, scalars, actions, masks, privileged = inputs()

    first = model(frames, scalars, actions, masks, privileged)
    second = model(frames, scalars, actions, masks, privileged + 100.0)

    assert first.logits.shape == (2, 3, 13)
    assert first.values.shape == (2, 3)
    assert first.hidden.shape == (1, 2, 256)
    assert torch.equal(first.logits, second.logits)
    assert not torch.equal(first.values, second.values)
    (first.logits.mean() + first.values.mean()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_sequence_mask_resets_hidden_before_marked_timestep() -> None:
    actor = RecurrentActor().eval()
    frames, scalars, actions, masks, _ = inputs(batch=1, time=3)
    hidden = torch.ones(1, 1, 256)
    masks[:, 1] = 0.0

    sequence = actor(frames, scalars, actions, masks, hidden)
    suffix = actor(
        frames[:, 1:],
        scalars[:, 1:],
        actions[:, 1:],
        torch.tensor([[0.0, 1.0]]),
        hidden,
    )

    torch.testing.assert_close(sequence.logits[:, 1:], suffix.logits)
    torch.testing.assert_close(sequence.hidden, suffix.hidden)


def test_actor_is_deterministic_traceable_without_critic_input() -> None:
    actor = RecurrentActor().eval()
    frames, scalars, actions, masks, _ = inputs(batch=1, time=2)
    hidden = actor.initial_state(1, device=frames.device)

    expected = actor(frames, scalars, actions, masks, hidden)
    repeated = actor(frames, scalars, actions, masks, hidden)
    traced = torch.jit.trace(actor, (frames, scalars, actions, masks, hidden))
    traced_logits, traced_features, traced_hidden = traced(
        frames, scalars, actions, masks, hidden
    )

    assert torch.equal(expected.logits, repeated.logits)
    torch.testing.assert_close(traced_logits, expected.logits)
    torch.testing.assert_close(traced_features, expected.features)
    torch.testing.assert_close(traced_hidden, expected.hidden)
    assert "privileged" not in str(traced.graph).lower()


def test_actor_accepts_uint8_dataset_frames() -> None:
    actor = RecurrentActor().eval()
    frames, scalars, actions, masks, _ = inputs(batch=1, time=1)
    uint8_frames = (frames * 255).to(torch.uint8)

    output = actor(uint8_frames, scalars, actions, masks)

    assert output.logits.shape == (1, 1, 13)
    assert torch.isfinite(output.logits).all()


@pytest.mark.parametrize(
    "field,replacement",
    (
        ("frames", torch.zeros(2, 3, 84, 84)),
        ("scalars", torch.zeros(2, 3, 5)),
        ("actions", torch.full((2, 3), 13)),
        ("masks", torch.full((2, 3), 0.5)),
    ),
)
def test_actor_rejects_invalid_inputs(field: str, replacement: torch.Tensor) -> None:
    actor = RecurrentActor()
    frames, scalars, actions, masks, _ = inputs()
    values = {"frames": frames, "scalars": scalars, "actions": actions, "masks": masks}
    values[field] = replacement

    with pytest.raises(ValueError):
        actor(values["frames"], values["scalars"], values["actions"], values["masks"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_cuda_inference_parity() -> None:
    torch.manual_seed(11)
    cpu = AsymmetricActorCritic().eval()
    cuda = AsymmetricActorCritic().cuda().eval()
    cuda.load_state_dict(cpu.state_dict())
    tensors = inputs(batch=1, time=2)

    with torch.no_grad():
        cpu_output = cpu(*tensors)
        cuda_output = cuda(*(tensor.cuda() for tensor in tensors))

    torch.testing.assert_close(
        cpu_output.logits, cuda_output.logits.cpu(), atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        cpu_output.values, cuda_output.values.cpu(), atol=2e-5, rtol=2e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_forward_backward_is_finite() -> None:
    model = AsymmetricActorCritic().cuda()
    tensors = inputs(batch=2, time=4, device="cuda")

    output = model(*tensors)
    loss = output.logits.square().mean() + output.values.square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
