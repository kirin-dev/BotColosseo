import pytest
import vizdoom as vzd

from botcolosseo.envs.actions import ACTION_BUTTONS, MacroAction, action_vector


def test_macro_action_ids_match_the_approved_contract() -> None:
    assert [(action.name, action.value) for action in MacroAction] == [
        ("IDLE", 0),
        ("MOVE_FORWARD", 1),
        ("MOVE_BACKWARD", 2),
        ("STRAFE_LEFT", 3),
        ("STRAFE_RIGHT", 4),
        ("TURN_LEFT", 5),
        ("TURN_RIGHT", 6),
        ("FORWARD_TURN_LEFT", 7),
        ("FORWARD_TURN_RIGHT", 8),
        ("ATTACK", 9),
        ("FORWARD_ATTACK", 10),
        ("TURN_LEFT_ATTACK", 11),
        ("TURN_RIGHT_ATTACK", 12),
    ]


def test_forward_attack_presses_exactly_two_buttons() -> None:
    vector = action_vector(MacroAction.FORWARD_ATTACK)

    assert len(vector) == len(ACTION_BUTTONS)
    assert vector[ACTION_BUTTONS.index(vzd.Button.MOVE_FORWARD)] == 1.0
    assert vector[ACTION_BUTTONS.index(vzd.Button.ATTACK)] == 1.0
    assert sum(vector) == 2.0


def test_every_macro_action_has_a_fixed_width_vector() -> None:
    assert all(len(action_vector(action)) == len(ACTION_BUTTONS) for action in MacroAction)


def test_invalid_macro_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown macro action"):
        action_vector(13)
