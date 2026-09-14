import torch

from botcolosseo.agents.hierarchical_critic import StrategicActorCritic
from botcolosseo.agents.hierarchical_model import StrategicActor
from botcolosseo.data.extraction_demonstrations import EXTRACTION_SCALAR_DIM


def test_privileged_value_isolated_from_fair_actor():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = StrategicActorCritic(StrategicActor())
    features = torch.randn(1, 2, 256, requires_grad=True)
    inputs = dict(
        features=features,
        scalars=torch.zeros(1, 2, EXTRACTION_SCALAR_DIM),
        previous_commands=torch.zeros(1, 2, dtype=torch.long),
        elapsed=torch.zeros(1, 2, 1),
        style=torch.zeros(1, 2, 3),
        difficulty=torch.ones(1, 2, 1),
        masks=torch.ones(1, 2),
    )
    first = model(privileged=torch.zeros(1, 2, 20), **inputs)
    changed = model(privileged=torch.ones(1, 2, 20), **inputs)
    torch.testing.assert_close(first.logits, changed.logits, rtol=0, atol=0)
    assert not torch.equal(first.values, changed.values)
    first.values.sum().backward()
    assert all(p.grad is None for p in model.actor.parameters())
    assert features.grad is None
    assert model.value[0].weight.grad is not None
