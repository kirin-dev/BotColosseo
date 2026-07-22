import pytest
import torch

from botcolosseo.training.rollout import RolloutBuffer, RolloutStep


def step(index: int, *, environments: int = 2) -> RolloutStep:
    return RolloutStep(
        frames=torch.full((environments, 1, 84, 84), index, dtype=torch.uint8),
        scalars=torch.full((environments, 6), float(index)),
        previous_actions=torch.full((environments,), index % 13, dtype=torch.long),
        masks=torch.ones(environments),
        privileged=torch.full((environments, 12), float(index)),
        hidden=torch.full((1, environments, 256), float(index)),
        actions=torch.full((environments,), (index + 1) % 13, dtype=torch.long),
        rewards=torch.ones(environments),
        terminated=torch.zeros(environments, dtype=torch.bool),
        truncated=torch.zeros(environments, dtype=torch.bool),
        log_probs=torch.full((environments,), -1.0),
        values=torch.full((environments,), 0.5),
        next_values=torch.full((environments,), 0.5),
    )


def test_rollout_buffer_finalizes_typed_time_axis() -> None:
    buffer = RolloutBuffer(capacity=3, environments=2)
    buffer.append(step(0))
    buffer.append(step(1))
    final = step(2)
    final.terminated[0] = True
    final.truncated[1] = True
    buffer.append(final)

    rollout = buffer.finalize(gamma=0.9, gae_lambda=0.8)

    assert rollout.frames.shape == (2, 3, 1, 84, 84)
    assert rollout.hidden.shape == (2, 3, 256)
    assert rollout.advantages.shape == (2, 3)
    assert rollout.valid.all()
    assert rollout.advantages[0, -1] == pytest.approx(0.5)
    assert rollout.advantages[1, -1] == pytest.approx(0.95)


def test_recurrent_sequences_exclude_burn_in_and_padding_from_loss() -> None:
    buffer = RolloutBuffer(capacity=5, environments=1)
    for index in range(5):
        current = step(index, environments=1)
        if index == 2:
            current.masks[0] = 0.0
        buffer.append(current)
    rollout = buffer.finalize(gamma=0.9, gae_lambda=0.8)

    batches = list(
        rollout.sequence_minibatches(
            sequence_length=2,
            burn_in=1,
            minibatch_sequences=1,
            seed=7,
            epoch=0,
            shuffle=False,
        )
    )

    assert len(batches) == 3
    assert batches[0].loss_mask.tolist() == [[True, True, False]]
    assert batches[1].loss_mask.tolist() == [[False, True, True]]
    assert batches[2].loss_mask.tolist() == [[False, True, False]]
    assert batches[1].frames[0, :, 0, 0, 0].tolist() == [1, 2, 3]
    assert batches[1].masks.tolist() == [[1.0, 0.0, 1.0]]
    assert batches[1].initial_hidden[0, 0, 0].item() == 1.0


def test_sequence_minibatch_shuffle_is_deterministic() -> None:
    buffer = RolloutBuffer(capacity=5, environments=2)
    for index in range(5):
        buffer.append(step(index))
    rollout = buffer.finalize(gamma=0.9, gae_lambda=0.8)

    def order() -> list[float]:
        batches = rollout.sequence_minibatches(
            sequence_length=2,
            burn_in=1,
            minibatch_sequences=2,
            seed=11,
            epoch=3,
            shuffle=True,
        )
        return [float(batch.initial_hidden[0, 0, 0]) for batch in batches]

    assert order() == order()


def test_rollout_rejects_overflow_and_nonfinite_values() -> None:
    buffer = RolloutBuffer(capacity=1, environments=2)
    invalid = step(0)
    invalid.rewards[0] = float("inf")
    with pytest.raises(FloatingPointError):
        buffer.append(invalid)

    buffer.append(step(0))
    with pytest.raises(OverflowError):
        buffer.append(step(1))


def test_rollout_owns_detached_snapshots() -> None:
    buffer = RolloutBuffer(capacity=1, environments=1)
    source = step(0, environments=1)
    source.log_probs.requires_grad_()
    buffer.append(source)
    source.rewards.fill_(99.0)

    rollout = buffer.finalize(gamma=0.9, gae_lambda=0.8)

    assert not rollout.log_probs.requires_grad
    assert rollout.returns[0, 0].item() == pytest.approx(1.45)
