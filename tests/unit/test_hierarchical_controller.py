import pytest
import torch

from botcolosseo.agents.hierarchical_controller import HierarchicalController
from botcolosseo.agents.hierarchical_model import CommandExecutor, StrategicActor
from botcolosseo.data.extraction_demonstrations import EXTRACTION_SCALAR_DIM
from botcolosseo.training.hierarchical_protocol import ControlCondition


def test_conditions_and_latched_event_preserve_memory():
    torch.set_num_threads(1)
    controller = HierarchicalController(CommandExecutor(), StrategicActor(), seed=17)
    inputs = (
        torch.zeros(1, 1, 1, 84, 84, dtype=torch.uint8),
        torch.zeros(1, 1, EXTRACTION_SCALAR_DIM),
        torch.zeros(1, 1, dtype=torch.long),
    )
    first = controller.step(*inputs, ControlCondition())
    assert first["replanned"]
    high_hidden = controller.high_hidden
    changed = ControlCondition(aggressive=1, difficulty=0.3)
    second = controller.step(*inputs, changed, public_event=True)
    assert not second["replanned"]
    assert second["style"] == (0, 0, 0)
    assert second["difficulty"] == 0.3
    assert controller.high_hidden is high_hidden
    third = controller.step(*inputs, changed)
    assert third["replanned"] and third["style"] == (1, 0, 0)
    snapshot = controller.last_high_inputs
    with torch.no_grad():
        replay = controller.strategy(**snapshot)
    assert float(replay.logits.log_softmax(-1)[0, 0, third["command"]]) == pytest.approx(
        third["command_log_prob"], abs=1e-7
    )
    assert snapshot["hidden"] is not high_hidden
    assert torch.equal(snapshot["hidden"], high_hidden)
    inputs[1].fill_(0.5)
    assert not snapshot["scalars"].any()  # No alias to the caller's mutable observation.
    assert controller.low_hidden is not None and controller.decisions == 3
    for _ in range(7):
        assert not controller.step(*inputs, changed)["replanned"]
        assert controller.last_high_inputs is None
    assert controller.step(*inputs, changed)["replanned"]
    controller.reset()
    assert controller.low_hidden is None and controller.high_hidden is None
    assert controller.decisions == 0
