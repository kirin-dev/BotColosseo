from __future__ import annotations

from enum import IntEnum

import vizdoom as vzd


class MacroAction(IntEnum):
    IDLE = 0
    MOVE_FORWARD = 1
    MOVE_BACKWARD = 2
    STRAFE_LEFT = 3
    STRAFE_RIGHT = 4
    TURN_LEFT = 5
    TURN_RIGHT = 6
    FORWARD_TURN_LEFT = 7
    FORWARD_TURN_RIGHT = 8
    ATTACK = 9
    FORWARD_ATTACK = 10
    TURN_LEFT_ATTACK = 11
    TURN_RIGHT_ATTACK = 12


ACTION_BUTTONS = (
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.ATTACK,
)

_ACTION_VECTORS: dict[MacroAction, tuple[float, ...]] = {
    MacroAction.IDLE: (0, 0, 0, 0, 0, 0, 0),
    MacroAction.MOVE_FORWARD: (1, 0, 0, 0, 0, 0, 0),
    MacroAction.MOVE_BACKWARD: (0, 1, 0, 0, 0, 0, 0),
    MacroAction.STRAFE_LEFT: (0, 0, 1, 0, 0, 0, 0),
    MacroAction.STRAFE_RIGHT: (0, 0, 0, 1, 0, 0, 0),
    MacroAction.TURN_LEFT: (0, 0, 0, 0, 1, 0, 0),
    MacroAction.TURN_RIGHT: (0, 0, 0, 0, 0, 1, 0),
    MacroAction.FORWARD_TURN_LEFT: (1, 0, 0, 0, 1, 0, 0),
    MacroAction.FORWARD_TURN_RIGHT: (1, 0, 0, 0, 0, 1, 0),
    MacroAction.ATTACK: (0, 0, 0, 0, 0, 0, 1),
    MacroAction.FORWARD_ATTACK: (1, 0, 0, 0, 0, 0, 1),
    MacroAction.TURN_LEFT_ATTACK: (0, 0, 0, 0, 1, 0, 1),
    MacroAction.TURN_RIGHT_ATTACK: (0, 0, 0, 0, 0, 1, 1),
}


def action_vector(action: MacroAction | int) -> list[float]:
    try:
        macro_action = MacroAction(action)
    except ValueError as exc:
        raise ValueError(f"Unknown macro action: {action}") from exc
    return [float(value) for value in _ACTION_VECTORS[macro_action]]
