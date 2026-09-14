"""One-player deployment state for high-sample / low-argmax inference."""

import torch

from botcolosseo.agents.hierarchical_model import CommandExecutor, StrategicActor
from botcolosseo.training.hierarchical_protocol import Command, ControlCondition, should_replan


class HierarchicalController:
    """No environment handle: only fair observation tensors and public events.

    Each player owns this state, even when executor weights are shared. The
    caller must reset at episode boundaries, not at condition changes.
    """

    def __init__(self, executor: CommandExecutor, strategy: StrategicActor, *, seed: int):
        self.executor = executor.eval()
        self.strategy = strategy.eval()
        self.device = next(executor.parameters()).device
        if next(strategy.parameters()).device != self.device:
            raise ValueError("Both actors must share a device")
        if strategy.feature_dim != executor.hidden_size:
            raise ValueError("Executor feature dimension mismatch")
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.reset()

    def reset(self):
        self.low_hidden = self.high_hidden = None
        self.features = torch.zeros(1, 1, self.executor.hidden_size, device=self.device)
        self.command = Command.SEARCH_CENTER
        self.elapsed = self.decisions = 0
        self.pending_event = False
        self.applied_condition = ControlCondition()
        self.requested_condition = ControlCondition()
        self.applied_low_difficulty = 1.0
        self.last_high_inputs = None

    def resolve_condition(self, condition: ControlCondition) -> ControlCondition:
        """Condition at the next decision, without advancing either recurrent state."""
        return condition

    @torch.no_grad()
    def step(
        self, frames, scalars, previous_actions, condition: ControlCondition, *, public_event=False
    ):
        if frames.shape[:2] != (1, 1):
            raise ValueError("Controller expects one player and one observation")
        self.requested_condition = condition
        self.applied_low_difficulty = condition.difficulty
        self.pending_event |= bool(public_event)
        replan = self.decisions == 0 or should_replan(self.elapsed, public_event=self.pending_event)
        command_log_prob = None
        self.last_high_inputs = None
        if replan:
            # Snapshot before changing command or advancing recurrent memory.
            self.last_high_inputs = {
                "features": self.features.detach().clone(),
                "scalars": scalars.detach().clone(),
                "previous_commands": torch.tensor([[int(self.command)]], device=self.device),
                "elapsed": torch.tensor([[[float(self.elapsed)]]], device=self.device),
                "style": torch.tensor(
                    [[condition.as_tuple()[:3]]], device=self.device, dtype=torch.float32
                ),
                "difficulty": torch.tensor(
                    [[[condition.difficulty]]], device=self.device, dtype=torch.float32
                ),
                "masks": torch.tensor([[float(self.decisions > 0)]], device=self.device),
                "hidden": None if self.high_hidden is None else self.high_hidden.detach().clone(),
            }
            output = self.strategy(**self.last_high_inputs)
            probabilities = output.logits[0, 0].softmax(-1)
            selected = torch.multinomial(probabilities, 1, generator=self.rng)
            self.command = Command(int(selected.item()))
            command_log_prob = float(output.logits[0, 0].log_softmax(-1)[selected].item())
            self.high_hidden = output.hidden
            self.applied_condition = condition
            self.elapsed = 0
            self.pending_event = False
        output = self.executor(
            frames,
            scalars,
            previous_actions,
            torch.tensor([[float(self.decisions > 0)]], device=self.device),
            torch.tensor([[int(self.command)]], device=self.device),
            torch.tensor([[[condition.difficulty]]], device=self.device, dtype=torch.float32),
            self.low_hidden,
        )
        self.low_hidden, self.features = output.hidden, output.features
        result = {
            "action": int(output.logits.argmax(-1).item()),
            "command": int(self.command),
            "replanned": replan,
            "command_log_prob": command_log_prob,
            "style": self.applied_condition.as_tuple()[:3],
            "difficulty": condition.difficulty,
            "high_difficulty": self.applied_condition.difficulty,
            "decision": self.decisions,
        }
        self.elapsed += 1
        self.decisions += 1
        return result
