import torch

from botcolosseo.agents.model import AsymmetricActorCritic
from botcolosseo.agents.style_model import RoutedStyledActorCritic, StyledActorCritic


def _inputs():
    return (
        torch.zeros(2, 3, 1, 84, 84, dtype=torch.uint8),
        torch.zeros(2, 3, 6),
        torch.zeros(2, 3, dtype=torch.long),
        torch.ones(2, 3),
        torch.zeros(2, 3, 12),
    )


def test_style_model_starts_identical_and_freezes_public_base() -> None:
    torch.manual_seed(7)
    base = AsymmetricActorCritic().eval()
    style = StyledActorCritic.from_base(base, bottleneck=32).eval()

    with torch.no_grad():
        expected = base(*_inputs())
        actual = style(*_inputs())

    torch.testing.assert_close(actual.logits, expected.logits)
    torch.testing.assert_close(actual.base_logits, expected.logits)
    assert all(not parameter.requires_grad for parameter in style.base.actor.parameters())
    assert all(parameter.requires_grad for parameter in style.adapter.parameters())
    assert all(parameter.requires_grad for parameter in style.policy.parameters())


def test_style_gradient_only_updates_adapter_policy_and_critic() -> None:
    style = StyledActorCritic.from_base(AsymmetricActorCritic(), bottleneck=16)
    output = style(*_inputs())
    (output.logits.mean() + output.values.mean()).backward()

    assert all(parameter.grad is None for parameter in style.base.actor.parameters())
    assert any(parameter.grad is not None for parameter in style.adapter.parameters())
    assert all(parameter.grad is not None for parameter in style.policy.parameters())
    assert all(
        parameter.grad is not None for parameter in style.base.privileged_encoder.parameters()
    )


def test_routed_style_has_independent_branches_over_one_frozen_base() -> None:
    torch.manual_seed(11)
    model = RoutedStyledActorCritic.from_base(
        AsymmetricActorCritic(), bottleneck=8
    ).eval()
    modes = torch.tensor([[0, 1, 2], [2, 1, 0]])

    with torch.no_grad():
        output = model(*_inputs(), route_modes=modes)
        base = model.base(*_inputs())

    torch.testing.assert_close(output.logits, base.logits)
    assert len(model.adapters) == 3
    assert model.adapters[0] is not model.adapters[1]
    assert model.policies[0] is not model.policies[1]
    assert all(
        not parameter.requires_grad for parameter in model.base.actor.parameters()
    )


def test_routed_style_selects_requested_branch_per_token() -> None:
    model = RoutedStyledActorCritic.from_base(
        AsymmetricActorCritic(), bottleneck=8
    ).eval()
    with torch.no_grad():
        for index, policy in enumerate(model.policies):
            policy.weight.zero_()
            policy.bias.fill_(float(index))
    modes = torch.tensor([[0, 1, 2], [2, 1, 0]])

    output = model(*_inputs(), route_modes=modes)

    assert output.logits[..., 0].tolist() == [
        [0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0],
    ]
