from __future__ import annotations

import torch

from botcolosseo.agents.extraction_governor import AggressiveCapabilityGovernor
from botcolosseo.agents.extraction_model import create_extraction_actor


def actor_with_action(action: int):
    actor = create_extraction_actor()
    actor.policy.weight.data.zero_()
    actor.policy.bias.data.zero_()
    actor.policy.bias.data[action] = 10
    return actor


def inputs(*, carried: int, health: int = 100, remaining: float = 75):
    frames = torch.zeros(1, 1, 1, 84, 84, dtype=torch.uint8)
    scalars = torch.zeros(1, 1, 9)
    scalars[0, 0, 0] = health / 100
    scalars[0, 0, 2] = carried / 150
    scalars[0, 0, 8] = remaining / 75
    previous = torch.zeros(1, 1, dtype=torch.long)
    masks = torch.zeros(1, 1)
    return frames, scalars, previous, masks


def test_aggressive_governor_latches_to_base_after_value_conversion() -> None:
    governor = AggressiveCapabilityGovernor(
        strong_base=actor_with_action(1),
        aggressive=actor_with_action(9),
    )
    hidden = governor.initial_state(1, device="cpu")

    early = governor(*inputs(carried=10), hidden)
    converted = governor(*inputs(carried=85, remaining=50), early.hidden)
    still_safe = governor(*inputs(carried=10, remaining=50), converted.hidden)

    assert int(early.logits.argmax(-1)) == 9
    assert int(converted.logits.argmax(-1)) == 1
    assert int(still_safe.logits.argmax(-1)) == 1
