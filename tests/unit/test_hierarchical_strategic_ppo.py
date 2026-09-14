import copy
import math

import pytest
import torch

from botcolosseo.agents.hierarchical_critic import StrategicActorCritic
from botcolosseo.agents.hierarchical_model import StrategicActor
from botcolosseo.data.extraction_demonstrations import EXTRACTION_SCALAR_DIM
from botcolosseo.training.hierarchical_strategic_ppo import update_strategic_ppo


@pytest.mark.parametrize("retention", [False, True])
def test_strategic_update_replays_memory_without_feature_gradient(retention):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = StrategicActorCritic(StrategicActor())
    inputs = dict(
        features=torch.randn(1, 4, 256, requires_grad=True),
        scalars=torch.zeros(1, 4, EXTRACTION_SCALAR_DIM),
        previous_commands=torch.zeros(1, 4, dtype=torch.long),
        elapsed=torch.zeros(1, 4, 1),
        style=torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]]),
        difficulty=torch.ones(1, 4, 1),
        masks=torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
    )
    privileged = torch.zeros(1, 4, 20)
    commands = torch.tensor([[0, 1, 2, 3]])
    with torch.no_grad():
        old = model(privileged=privileged, **inputs)
    batch = dict(
        **inputs,
        privileged=privileged,
        commands=commands,
        old_log_probs=torch.distributions.Categorical(logits=old.logits).log_prob(commands),
        old_values=old.values,
        next_values=torch.cat((old.values[:, 1:], torch.zeros(1, 1)), 1),
        rewards=torch.tensor([[0.0, 0.0, 0.0, 0.5]]),
        durations=torch.tensor([[8, 2, 8, 3]]),
        terminated=torch.tensor([[False, False, False, True]]),
        truncated=torch.zeros(1, 4, dtype=torch.bool),
    )
    before = model.actor.policy.weight.detach().clone()
    reference = copy.deepcopy(model.actor).requires_grad_(False)
    reference_before = {k: v.clone() for k, v in reference.state_dict().items()}
    result = update_strategic_ppo(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-4),
        batch,
        chunk_size=2,
        reference=reference if retention else None,
        reference_weight=0.1 if retention else 0,
    )
    assert all(math.isfinite(v) for v in result.values())
    assert abs(result["approximate_kl"]) < 1e-6
    assert not torch.equal(before, model.actor.policy.weight)
    assert inputs["features"].grad is None
    assert all(torch.equal(v, reference.state_dict()[k]) for k, v in reference_before.items())
    assert abs(result["reference_kl"]) < 1e-6
