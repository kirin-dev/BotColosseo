import torch

from botcolosseo.agents.hierarchical_model import CommandExecutor, StrategicActor
from botcolosseo.data.extraction_demonstrations import EXTRACTION_SCALAR_DIM
from botcolosseo.demo.hierarchical_controls import ScheduledController, control_schedule
from botcolosseo.demo.style_response_probe import ProbedScheduledController
from botcolosseo.training.hierarchical_protocol import ControlCondition


def test_read_only_probe_preserves_actions_memory_and_sampling_state():
    torch.set_num_threads(1)
    low, high = CommandExecutor(), StrategicActor()
    schedule = control_schedule(single_style="defensive")
    plain = ScheduledController(low, high, seed=1701, schedule=schedule)
    probed = ProbedScheduledController(low, high, seed=1701, schedule=schedule)
    inputs = (torch.zeros(1, 1, 1, 84, 84, dtype=torch.uint8),
              torch.zeros(1, 1, EXTRACTION_SCALAR_DIM), torch.zeros(1, 1, dtype=torch.long))
    for _ in range(90):
        assert plain.step(*inputs, ControlCondition()) == probed.step(*inputs, ControlCondition())
        assert torch.equal(plain.rng.get_state(), probed.rng.get_state())
        assert torch.equal(plain.high_hidden, probed.high_hidden)
        assert torch.equal(plain.low_hidden, probed.low_hidden)
    assert probed.style_probes
    assert set(probed.style_probes[-1]["command_probabilities"]) == {
        "neutral", "aggressive", "defensive", "explorer"}
    assert probed.applied_condition.defensive == 1
    probed.reset()
    assert probed.style_probes == []
