import copy

import pytest
import torch

from botcolosseo.agents.hierarchical_model import CommandExecutor, StrategicActor
from botcolosseo.cli.prepare_hierarchical_control_candidate import prepare
from botcolosseo.cli.train_hierarchical_strategic import digest


def test_candidate_preserves_high_actor_and_rejects_reference_drift(tmp_path):
    base = CommandExecutor()
    reference = tmp_path / "reference.pt"
    torch.save({"actor": base.state_dict()}, reference)
    actor = copy.deepcopy(base)
    actor.difficulty_input.anchor_at_hard()
    actor.difficulty_output.anchor_at_hard()
    candidate = tmp_path / "candidate.pt"
    payload = {
        "actor": actor.state_dict(),
        "identity": {
            "difficulty_distillation": {
                "anchor_hard": True,
                "teachers": ["easy", "normal", digest(reference)],
            }
        },
    }
    torch.save(payload, candidate)
    high = tmp_path / "high.pt"
    high_state = StrategicActor().state_dict()
    torch.save({"actor": high_state, "identity": {"executor": digest(reference)}}, high)
    result = prepare(candidate, reference, [high], tmp_path / "bundle")
    saved = torch.load(result["strategies"][0]["candidate"], weights_only=False)
    assert saved["identity"]["executor"] == digest(candidate)
    assert all(torch.equal(v, saved["actor"][k]) for k, v in high_state.items())
    payload["actor"]["difficulty_input.anchor_network.0.weight"].add_(1)
    torch.save(payload, candidate)
    with pytest.raises(ValueError, match="anchor differs"):
        prepare(candidate, reference, [high], tmp_path / "bad-bundle")
    assert not (tmp_path / "bad-bundle").exists()
