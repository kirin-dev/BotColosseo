"""Frozen high-level command selection during shared-executor optimization."""

import numpy as np
import torch

from botcolosseo.training.hierarchical_protocol import Command, should_replan


def upgrade_pair(solution, episode):
    """Balanced uniform/meta source blocks, independently sampled role identities."""
    source = "uniform" if (episode // 2) % 2 == 0 else "meta"
    rng = np.random.default_rng(4701 + episode)
    selected = {}
    for role in ("host", "opponent"):
        marginal = np.asarray(solution[role], dtype=float)
        if (
            marginal.ndim != 1
            or not 1 <= len(marginal) <= 6
            or not np.isfinite(marginal).all()
            or (marginal < 0).any()
            or not np.isclose(marginal.sum(), 1)
        ):
            raise ValueError("Upgrade requires valid role meta-strategies")
        probabilities = np.ones(len(marginal)) / len(marginal) if source == "uniform" else marginal
        selected[role] = int(rng.choice(len(marginal), p=probabilities))
    return source, selected


class FrozenCommandSelector:
    """Consume only the actual preceding executor memory and fair own scalars.

    One call per low decision, including the timeout bootstrap decision. No
    privileged scorer or alternate executor supplies strategic observations.
    """

    def __init__(self, strategy, *, seed):
        if any(parameter.requires_grad for parameter in strategy.parameters()):
            raise ValueError("Executor upgrades require a frozen strategic actor")
        self.strategy = strategy.eval()
        self.device = next(strategy.parameters()).device
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.hidden = None
        self.command = Command.SEARCH_CENTER
        self.elapsed = self.decisions = 0
        self.pending_event = False

    @torch.no_grad()
    def select(self, low_hidden, scalars, *, public_event=False):
        self.pending_event |= bool(public_event)
        if self.decisions == 0 or should_replan(self.elapsed, public_event=self.pending_event):
            features = (
                torch.zeros(1, 1, self.strategy.feature_dim, device=self.device)
                if low_hidden is None
                else low_hidden.transpose(0, 1).detach()
            )
            output = self.strategy(
                features=features,
                scalars=scalars,
                previous_commands=torch.tensor([[int(self.command)]], device=self.device),
                elapsed=torch.tensor([[[float(self.elapsed)]]], device=self.device),
                style=torch.zeros(1, 1, 3, device=self.device),
                difficulty=torch.ones(1, 1, 1, device=self.device),
                masks=torch.tensor([[float(self.decisions > 0)]], device=self.device),
                hidden=self.hidden,
            )
            self.command = Command(
                int(
                    torch.multinomial(output.logits[0, 0].softmax(-1), 1, generator=self.rng).item()
                )
            )
            self.hidden = output.hidden
            self.elapsed, self.pending_event = 0, False
        self.elapsed += 1
        self.decisions += 1
        return self.command
