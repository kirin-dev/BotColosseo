import math
from pathlib import Path

import pytest
import torch

from botcolosseo.agents.model import AsymmetricActorCritic
from botcolosseo.training.ppo import (
    PPOBatch,
    PPOTrainer,
    clipped_ppo_loss,
    ppo_loss,
)


def test_clipped_policy_and_value_losses_match_hand_calculation() -> None:
    metrics = clipped_ppo_loss(
        new_log_probs=torch.log(torch.tensor([1.3, 0.7])),
        entropy=torch.tensor([0.5, 0.5]),
        values=torch.tensor([2.0, -1.0]),
        old_log_probs=torch.zeros(2),
        old_values=torch.zeros(2),
        advantages=torch.tensor([1.0, -1.0]),
        returns=torch.tensor([1.0, -1.0]),
        valid=torch.ones(2, dtype=torch.bool),
        policy_clip=0.2,
        value_clip=0.5,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        max_kl=1.0,
    )

    assert metrics.policy_loss.item() == pytest.approx(-0.2)
    assert metrics.value_loss.item() == pytest.approx(0.3125)
    assert metrics.entropy.item() == pytest.approx(0.5)
    assert metrics.total_loss.item() == pytest.approx(-0.04875)
    assert metrics.clip_fraction == pytest.approx(1.0)


def test_entropy_bonus_lowers_minimized_loss() -> None:
    common = dict(
        values=torch.zeros(1),
        actions=torch.zeros(1, dtype=torch.long),
        old_log_probs=torch.tensor([math.log(0.5)]),
        old_values=torch.zeros(1),
        advantages=torch.zeros(1),
        returns=torch.zeros(1),
        valid=torch.ones(1, dtype=torch.bool),
        policy_clip=0.2,
        value_clip=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.1,
        max_kl=10.0,
    )
    uniform = ppo_loss(logits=torch.zeros(1, 2), **common)
    confident = ppo_loss(logits=torch.tensor([[8.0, -8.0]]), **common)

    assert uniform.entropy > confident.entropy
    assert uniform.total_loss < confident.total_loss


def test_ppo_rejects_nonfinite_values_and_excessive_kl() -> None:
    arguments = dict(
        new_log_probs=torch.zeros(1),
        entropy=torch.zeros(1),
        values=torch.zeros(1),
        old_log_probs=torch.zeros(1),
        old_values=torch.zeros(1),
        advantages=torch.ones(1),
        returns=torch.zeros(1),
        valid=torch.ones(1, dtype=torch.bool),
        policy_clip=0.2,
        value_clip=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        max_kl=0.1,
    )
    arguments["values"] = torch.tensor([float("nan")])
    with pytest.raises(FloatingPointError):
        clipped_ppo_loss(**arguments)

    arguments["values"] = torch.zeros(1)
    arguments["new_log_probs"] = torch.tensor([-10.0])
    with pytest.raises(RuntimeError, match="KL"):
        clipped_ppo_loss(**arguments)


def synthetic_batch(model: AsymmetricActorCritic) -> PPOBatch:
    frames = torch.zeros(1, 2, 1, 84, 84, dtype=torch.uint8)
    scalars = torch.zeros(1, 2, 6)
    previous_actions = torch.zeros(1, 2, dtype=torch.long)
    masks = torch.tensor([[0.0, 1.0]])
    privileged = torch.zeros(1, 2, 12)
    initial_hidden = torch.zeros(1, 1, 256)
    with torch.no_grad():
        output = model(
            frames,
            scalars,
            previous_actions,
            masks,
            privileged,
            initial_hidden,
        )
        distribution = torch.distributions.Categorical(logits=output.logits)
        actions = output.logits.argmax(dim=-1)
        old_log_probs = distribution.log_prob(actions)
    return PPOBatch(
        frames=frames,
        scalars=scalars,
        previous_actions=previous_actions,
        masks=masks,
        privileged=privileged,
        initial_hidden=initial_hidden,
        actions=actions,
        old_log_probs=old_log_probs,
        old_values=output.values,
        advantages=torch.ones(1, 2),
        returns=output.values + 1.0,
        loss_mask=torch.ones(1, 2, dtype=torch.bool),
    )


def test_synthetic_update_lowers_loss_and_resumes_exactly(tmp_path: Path) -> None:
    torch.manual_seed(23)
    model = AsymmetricActorCritic()
    batch = synthetic_batch(model)
    trainer = PPOTrainer.create(
        model,
        learning_rate=1e-4,
        total_updates=10,
        gradient_clip=0.5,
        policy_clip=0.2,
        value_clip=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.0,
        max_kl=1.0,
    )

    initial = trainer.evaluate(batch).total_loss
    trainer.train_step(batch)
    after = trainer.evaluate(batch).total_loss
    checkpoint = trainer.save(
        tmp_path / "ppo.pt", config_hash="config", scenario_hash="scenario"
    )
    baseline = trainer.train_step(batch)
    baseline_state = {
        name: value.detach().clone() for name, value in trainer.model.state_dict().items()
    }
    resumed = PPOTrainer.create(
        AsymmetricActorCritic(),
        learning_rate=1e-4,
        total_updates=10,
        gradient_clip=0.5,
        policy_clip=0.2,
        value_clip=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.0,
        max_kl=1.0,
    )
    resumed.load(
        checkpoint,
        config_hash="config",
        scenario_hash="scenario",
        restore_rng=True,
    )
    repeated = resumed.train_step(batch)

    assert after < initial
    assert repeated.total_loss == baseline.total_loss
    assert resumed.updates == trainer.updates
    assert all(
        torch.equal(value, baseline_state[name])
        for name, value in resumed.model.state_dict().items()
    )
