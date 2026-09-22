from types import SimpleNamespace

import numpy as np
import pytest
import torch

from botcolosseo.cli import probe_hierarchical_search as probe
from botcolosseo.training.hierarchical_protocol import Command


@pytest.mark.parametrize("difficulty", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_difficulty_rejected_before_environment(difficulty):
    with pytest.raises(ValueError, match="Difficulty"):
        probe.trial(None, seed=0, side="host", command=Command.SEARCH_SOUTH,
                    scenario="test", device="cpu", difficulty=difficulty)


@pytest.mark.parametrize("health,banked", [(0, 0), (100, 50)])
def test_probe_records_inactive_cause_and_applied_difficulty(monkeypatch, health, banked):
    class Env:
        def __init__(self, **kwargs):
            self.done = False

        def observation(self):
            own = SimpleNamespace(
                frame=np.zeros((84, 84), dtype=np.uint8), previous_action=0,
                health=health if self.done else 100,
                banked_value=banked if self.done else 0,
            )
            return SimpleNamespace(host=own, terminated=self.done, truncated=False)

        def reset(self):
            return self.observation(), SimpleNamespace(scenario_hash="test")

        def privileged_state(self):
            return self.done

        def protocol_snapshot(self):
            return None

        def step(self, *args):
            self.done = True
            return self.observation()

        def close(self):
            pass

    class Teacher:
        def __init__(self, **kwargs):
            pass

        def begin(self, *args):
            pass

        def act(self, done):
            return SimpleNamespace(status="inactive" if done else "executing")

        def observe_transition(self, *args):
            pass

    seen = []

    def actor(*args):
        seen.append(args[5].item())
        return SimpleNamespace(hidden=None, logits=torch.zeros(1, 1, 13))

    monkeypatch.setattr(probe, "SynchronousExtractionEnv", Env)
    monkeypatch.setattr(probe, "PrivilegedCommandTeacher", Teacher)
    monkeypatch.setattr(probe, "PrivilegedStrongExtractionTeacher",
                        lambda **kw: SimpleNamespace(act=lambda state: 0))
    monkeypatch.setattr(probe, "extraction_scalars", lambda obs: np.zeros(9, dtype=np.float32))
    result = probe.trial(actor, seed=0, side="host", command=Command.SEARCH_SOUTH,
                         scenario="test", device="cpu", difficulty=0.5)
    assert seen == [0.5]
    assert result["final_health"] == health
    assert result["final_banked"] == banked
    assert result["difficulty"] == 0.5
    assert result["terminated"] and not result["success"]
    assert result["outcome"] == "inactive"
