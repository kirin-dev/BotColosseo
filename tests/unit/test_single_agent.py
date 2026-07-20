import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import vizdoom as vzd

from botcolosseo.envs.actions import ACTION_BUTTONS, MacroAction
from botcolosseo.envs.events import EventType
from botcolosseo.envs.single_agent import SingleAgentTaskEnv
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.scenarios.splits import TaskKind


class FakeGame:
    def __init__(self, *, missing_state: bool = False, raise_on_action: bool = False) -> None:
        self.missing_state = missing_state
        self.raise_on_action = raise_on_action
        self.closed = False
        self.finished = False
        self.new_episode_calls = 0
        self.actions: list[tuple[list[float], int]] = []
        self.user_values = [1, 10, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def new_episode(self) -> None:
        self.new_episode_calls += 1

    def get_state(self):
        if self.missing_state or self.finished:
            return None
        return SimpleNamespace(screen_buffer=np.zeros((240, 320), dtype=np.uint8))

    def get_game_variable(self, variable: vzd.GameVariable) -> float:
        if variable.name.startswith("USER"):
            return float(self.user_values[int(variable.name[4:]) - 1])
        return {
            vzd.GameVariable.HEALTH: 100.0,
            vzd.GameVariable.SELECTED_WEAPON_AMMO: 50.0,
            vzd.GameVariable.ATTACK_READY: 1.0,
            vzd.GameVariable.POSITION_X: -640.0,
            vzd.GameVariable.POSITION_Y: 0.0,
            vzd.GameVariable.ANGLE: 0.0,
        }.get(variable, 0.0)

    def make_action(self, action: list[float], frame_skip: int) -> float:
        if self.raise_on_action:
            raise RuntimeError("engine failed")
        self.actions.append((action, frame_skip))
        self.user_values[1] += frame_skip
        self.user_values[11] = 1
        self.finished = True
        return 0.0

    def is_episode_finished(self) -> bool:
        return self.finished

    def close(self) -> None:
        self.closed = True


def make_env(tmp_path: Path, fake: FakeGame, *, max_decisions: int = 5):
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    config_path = scenario_dir / "crystal_run.cfg"
    config_path.write_text("doom_map = map01\n", encoding="utf-8")
    (scenario_dir / "manifest.json").write_text(
        json.dumps({"wad_sha256": "abc123"}),
        encoding="utf-8",
    )
    captured = []

    def builder(settings):
        captured.append(settings)
        return fake

    graph = RegionGraph.from_yaml(Path("assets/scenarios/crystal_run/src/regions.yaml"))
    env = SingleAgentTaskEnv(
        config_path=config_path,
        region_graph=graph,
        max_decisions=max_decisions,
        game_builder=builder,
    )
    return env, captured


def test_reset_selects_task_map_and_returns_only_legal_observation(tmp_path: Path) -> None:
    fake = FakeGame()
    env, captured = make_env(tmp_path, fake)

    observation, info = env.reset(seed=17, task=TaskKind.NAVIGATION)

    assert captured[0].seed == 17
    assert captured[0].doom_map == "MAP02"
    assert fake.new_episode_calls == 1
    assert observation.frame.shape == (84, 84)
    assert observation.health == 100.0
    assert info.scenario_hash == "abc123"
    assert info.task is TaskKind.NAVIGATION


def test_step_maps_action_decodes_success_and_uses_last_frame(tmp_path: Path) -> None:
    fake = FakeGame()
    env, unused = make_env(tmp_path, fake)
    env.reset(seed=17, task=TaskKind.NAVIGATION)

    step = env.step(MacroAction.FORWARD_ATTACK)

    vector, frame_skip = fake.actions[0]
    assert frame_skip == 4
    assert vector[ACTION_BUTTONS.index(vzd.Button.MOVE_FORWARD)] == 1.0
    assert vector[ACTION_BUTTONS.index(vzd.Button.ATTACK)] == 1.0
    assert step.terminated
    assert not step.truncated
    assert [event.type for event in step.events] == [EventType.TASK_SUCCESS]
    assert step.observation.frame.shape == (84, 84)


def test_decision_limit_is_truncation_not_termination(tmp_path: Path) -> None:
    fake = FakeGame()
    original_make_action = fake.make_action

    def nonterminal_action(action, frame_skip):
        result = original_make_action(action, frame_skip)
        fake.finished = False
        fake.user_values[11] = 0
        return result

    fake.make_action = nonterminal_action
    env, unused = make_env(tmp_path, fake, max_decisions=1)
    env.reset(seed=17, task=TaskKind.NAVIGATION)

    step = env.step(MacroAction.IDLE)

    assert not step.terminated
    assert step.truncated


@pytest.mark.parametrize("missing_state,raise_on_action", [(True, False), (False, True)])
def test_engine_failures_close_the_game(
    tmp_path: Path,
    missing_state: bool,
    raise_on_action: bool,
) -> None:
    fake = FakeGame(missing_state=missing_state, raise_on_action=raise_on_action)
    env, unused = make_env(tmp_path, fake)

    with pytest.raises(RuntimeError):
        if missing_state:
            env.reset(seed=17, task=TaskKind.NAVIGATION)
        else:
            env.reset(seed=17, task=TaskKind.NAVIGATION)
            env.step(MacroAction.IDLE)

    assert fake.closed
