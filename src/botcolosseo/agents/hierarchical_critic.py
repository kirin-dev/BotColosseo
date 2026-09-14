"""Training-only strategic value baseline with an isolated fair command actor."""

import torch
from torch import nn

from botcolosseo.agents.hierarchical_model import StrategicActor
from botcolosseo.agents.model import ActorCriticOutput


class StrategicActorCritic(nn.Module):
    def __init__(self, actor: StrategicActor, *, privileged_dim: int = 20):
        super().__init__()
        if privileged_dim <= 0:
            raise ValueError("Privileged dimension must be positive")
        self.actor = actor
        self.privileged_dim = privileged_dim
        self.value = nn.Sequential(
            nn.Linear(actor.hidden_size + privileged_dim + 4, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, *, privileged: torch.Tensor, **fair_inputs) -> ActorCriticOutput:
        # Populated recurrent features are from the fair actor only. Conditions
        # are explicit because actor.features precedes output FiLM modulation.
        output = self.actor(**fair_inputs)
        if privileged.shape != (*output.features.shape[:2], self.privileged_dim):
            raise ValueError("Invalid strategic privileged shape")
        if privileged.device != output.features.device or not privileged.is_floating_point():
            raise ValueError("Invalid strategic privileged device/dtype")
        if not bool(torch.isfinite(privileged).all()):
            raise ValueError("Nonfinite strategic privileged state")
        critic_input = torch.cat(
            (
                output.features.detach(),
                privileged,
                fair_inputs["style"].detach(),
                fair_inputs["difficulty"].detach(),
            ),
            -1,
        )
        values = self.value(critic_input).squeeze(-1)
        return ActorCriticOutput(output.logits, values, output.hidden)
