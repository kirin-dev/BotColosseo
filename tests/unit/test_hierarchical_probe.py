from types import SimpleNamespace

import numpy as np
import pytest

from botcolosseo.cli import probe_hierarchical_teacher as probe
from botcolosseo.training.hierarchical_protocol import Command


@pytest.mark.parametrize("sustained", [False, True])
def test_contact_does_not_end_sustained_combat(monkeypatch, tmp_path, sustained):
    state = SimpleNamespace(host_banked=0, opponent_banked=0)
    observation = SimpleNamespace(frame=np.zeros((84, 84), dtype=np.uint8), previous_action=0)

    class Env:
        def __init__(self, **kwargs):
            self.steps = 0

        def reset(self):
            return SimpleNamespace(host=observation), SimpleNamespace(scenario_hash="test")

        def privileged_state(self):
            return state

        def protocol_snapshot(self):
            return None

        def step(self, *actions):
            self.steps += 1
            return SimpleNamespace(host=observation, terminated=self.steps == 8, truncated=False)

        def close(self):
            pass

    class Teacher:
        def __init__(self, **kwargs):
            self.command = None

        def begin(self, command, state):
            self.command = command

        def act(self, state):
            return SimpleNamespace(
                action=0,
                status="unavailable" if self.command.value < 3 else "executing",
            )

        def observe_transition(self, *args):
            pass

    monkeypatch.setattr(probe, "SynchronousExtractionEnv", Env)
    monkeypatch.setattr(probe, "PrivilegedCommandTeacher", Teacher)
    monkeypatch.setattr(
        probe, "PrivilegedStrongExtractionTeacher", lambda **kw: SimpleNamespace(act=lambda s: 0)
    )
    monkeypatch.setattr(probe, "player_pose", lambda *args: (0, 0, 0))
    monkeypatch.setattr(probe, "opponent_position", lambda *args: (100, 0))
    monkeypatch.setattr(probe, "extraction_scalars", lambda obs: np.zeros(9, dtype=np.float32))
    path = tmp_path / "episode.npz"
    result = probe.run_case(
        33, "host", tmp_path / "unused.cfg", trajectory=path, sustained_combat=sustained
    )
    engage = next(r for r in result["commands"] if r["command"] == Command.ENGAGE.name)
    assert engage["decisions"] == (8 if sustained else 2)
    with np.load(path) as episode:
        assert bool(episode["sustained_combat"]) == sustained
