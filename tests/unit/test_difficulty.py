from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

from botcolosseo.agents.difficulty import (
    DifficultyPolicy,
    DifficultyProfile,
    load_difficulty_profiles,
)
from botcolosseo.envs.actions import MacroAction


class SequencePolicy:
    def __init__(self, actions: Sequence[int | MacroAction]) -> None:
        self.actions = iter(actions)
        self.observations: list[object] = []
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def act(self, observation: object) -> MacroAction:
        self.observations.append(observation)
        return MacroAction(next(self.actions))


def test_difficulty_policy_applies_exact_delay_and_update_cadence() -> None:
    base = SequencePolicy((MacroAction.MOVE_FORWARD, MacroAction.ATTACK))
    policy = DifficultyPolicy(
        base,
        DifficultyProfile(reaction_delay=2, policy_update_interval=2),
    )
    observations = [object() for _ in range(4)]

    policy.reset()
    actions = [policy.act(observation) for observation in observations]

    assert actions == [
        MacroAction.IDLE,
        MacroAction.IDLE,
        MacroAction.MOVE_FORWARD,
        MacroAction.MOVE_FORWARD,
    ]
    assert base.observations == [observations[0], observations[2]]


def test_difficulty_reset_clears_held_and_delayed_actions() -> None:
    base = SequencePolicy((MacroAction.ATTACK, MacroAction.MOVE_FORWARD))
    policy = DifficultyPolicy(
        base,
        DifficultyProfile(reaction_delay=1, policy_update_interval=1),
    )

    policy.reset()
    assert policy.act(object()) is MacroAction.IDLE
    policy.reset()

    assert policy.act(object()) is MacroAction.IDLE
    assert base.reset_count == 2


def test_hard_difficulty_is_the_native_action_stream() -> None:
    actions = (
        MacroAction.MOVE_FORWARD,
        MacroAction.TURN_LEFT,
        MacroAction.ATTACK,
    )
    policy = DifficultyPolicy(
        SequencePolicy(actions),
        DifficultyProfile(reaction_delay=0, policy_update_interval=1),
    )

    policy.reset()

    assert tuple(policy.act(object()) for _ in actions) == actions


def test_difficulty_policy_rejects_invalid_actions_and_requires_reset() -> None:
    policy = DifficultyPolicy(
        SequencePolicy((999,)),
        DifficultyProfile(reaction_delay=0, policy_update_interval=1),
    )

    with pytest.raises(RuntimeError, match="reset"):
        policy.act(object())
    policy.reset()
    with pytest.raises(ValueError, match="invalid action"):
        policy.act(object())


def test_frozen_difficulty_config_is_ordered() -> None:
    root = Path(__file__).resolve().parents[2]

    profiles = load_difficulty_profiles(root / "configs/difficulty.yaml")

    assert profiles == {
        "easy": DifficultyProfile(2, 2),
        "normal": DifficultyProfile(1, 1),
        "hard": DifficultyProfile(0, 1),
    }


def test_difficulty_config_rejects_non_native_hard(tmp_path: Path) -> None:
    path = tmp_path / "difficulty.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "profiles": {
                    "easy": {
                        "reaction_delay": 2,
                        "policy_update_interval": 2,
                    },
                    "normal": {
                        "reaction_delay": 1,
                        "policy_update_interval": 1,
                    },
                    "hard": {
                        "reaction_delay": 1,
                        "policy_update_interval": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="native"):
        load_difficulty_profiles(path)
