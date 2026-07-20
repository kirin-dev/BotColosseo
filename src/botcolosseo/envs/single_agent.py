from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import vizdoom as vzd

from botcolosseo.envs.actions import MacroAction, action_vector
from botcolosseo.envs.events import EventDecoder, EventType, ProtocolSnapshot
from botcolosseo.envs.rewards import RewardLedger
from botcolosseo.envs.types import ActorObservation, PrivilegedState, TaskStep
from botcolosseo.envs.vizdoom_game import GameSettings, create_game
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.scenarios.splits import TaskKind, TaskVariant, load_task_variants

_USER_VARIABLES = tuple(getattr(vzd.GameVariable, f"USER{index}") for index in range(1, 21))
_TARGET_REGIONS = {
    TaskKind.NAVIGATION: "center",
    TaskKind.PICKUP: "center",
    TaskKind.RETURN: "home",
    TaskKind.STATIC_HIT: "shooting_lane",
    TaskKind.MOVING_HIT: "shooting_lane",
}


@dataclass(frozen=True)
class ResetInfo:
    episode_id: int
    seed: int
    task: TaskKind
    scenario_hash: str


class SingleAgentTaskEnv:
    def __init__(
        self,
        *,
        config_path: Path,
        region_graph: RegionGraph,
        frame_skip: int = 4,
        max_decisions: int = 225,
        game_builder: Callable[[GameSettings], Any] = create_game,
        variants_path: Path | None = None,
    ) -> None:
        if frame_skip <= 0 or max_decisions <= 0:
            raise ValueError("frame_skip and max_decisions must be positive")
        self._config_path = config_path.expanduser().resolve()
        self._graph = region_graph
        self._frame_skip = frame_skip
        self._max_decisions = max_decisions
        self._game_builder = game_builder
        if variants_path is None:
            repository_root = Path(__file__).resolve().parents[3]
            variants_path = repository_root / "assets/scenarios/crystal_run/src/task_variants.yaml"
        self._variants = load_task_variants(variants_path)
        self._scenario_hash = self._load_scenario_hash()
        self._decoder = EventDecoder()
        self._ledger = RewardLedger(region_graph)
        self._game: Any | None = None
        self._task: TaskKind | None = None
        self._variant: TaskVariant | None = None
        self._episode_id = -1
        self._decision_index = 0
        self._last_observation: ActorObservation | None = None
        self._last_privileged: PrivilegedState | None = None

    def reset(self, *, seed: int, task: TaskKind) -> tuple[ActorObservation, ResetInfo]:
        self.close()
        variant = self._variants[TaskKind(task)]
        self._episode_id += 1
        self._decision_index = 0
        self._task = TaskKind(task)
        self._variant = variant
        self._decoder.reset()
        self._ledger.reset()
        game = self._game_builder(
            GameSettings(
                config_path=self._config_path,
                seed=seed,
                doom_map=variant.map_name,
            )
        )
        self._game = game
        try:
            game.new_episode()
            state = game.get_state()
            if state is None:
                raise RuntimeError("ViZDoom returned no state after task reset")
            snapshot, privileged = self._read_protocol_and_privileged(game)
            self._decoder.decode(
                snapshot,
                region_name=privileged.region_name,
                episode_id=self._episode_id,
                decision_index=0,
            )
            observation = self._make_observation(state.screen_buffer, snapshot, game)
            self._last_observation = observation
            self._last_privileged = privileged
            return observation, ResetInfo(
                episode_id=self._episode_id,
                seed=seed,
                task=self._task,
                scenario_hash=self._scenario_hash,
            )
        except BaseException:
            self.close()
            raise

    def step(self, action: MacroAction | int) -> TaskStep:
        game = self._require_game()
        if self._task is None or self._variant is None or self._last_observation is None:
            raise RuntimeError("Environment must be reset before step")
        macro_action = MacroAction(action)
        try:
            game.make_action(action_vector(macro_action), self._frame_skip)
            self._decision_index += 1
            snapshot, privileged = self._read_protocol_and_privileged(game)
            events = self._decoder.decode(
                snapshot,
                region_name=privileged.region_name,
                episode_id=self._episode_id,
                decision_index=self._decision_index,
            )
            reward = self._ledger.apply(
                events,
                target_region=_TARGET_REGIONS[self._task],
            )
            state = game.get_state()
            frame = (
                state.screen_buffer
                if state is not None
                else self._last_observation.frame
            )
            observation = self._make_observation(frame, snapshot, game, macro_action)
            terminated = any(event.type is EventType.TASK_SUCCESS for event in events)
            truncated = not terminated and (
                self._decision_index >= self._max_decisions or game.is_episode_finished()
            )
            self._last_observation = observation
            self._last_privileged = privileged
            return TaskStep(observation, reward, terminated, truncated, events)
        except BaseException:
            self.close()
            raise

    def teacher_state(self) -> PrivilegedState:
        if self._last_privileged is None:
            raise RuntimeError("Environment must be reset before requesting Teacher state")
        return self._last_privileged

    def close(self) -> None:
        if self._game is not None:
            self._game.close()
            self._game = None

    def _read_protocol_and_privileged(
        self,
        game: Any,
    ) -> tuple[ProtocolSnapshot, PrivilegedState]:
        values = [game.get_game_variable(variable) for variable in _USER_VARIABLES]
        snapshot = ProtocolSnapshot.from_values(values)
        player_x = float(game.get_game_variable(vzd.GameVariable.POSITION_X))
        player_y = float(game.get_game_variable(vzd.GameVariable.POSITION_Y))
        region = self._graph.region_at(player_x, player_y)
        privileged = PrivilegedState(
            player_x=player_x,
            player_y=player_y,
            player_angle=float(game.get_game_variable(vzd.GameVariable.ANGLE)),
            region_name=region.name if region is not None else None,
            core_x=snapshot.core_x,
            core_y=snapshot.core_y,
            target_x=snapshot.target_x,
            target_y=snapshot.target_y,
            target_alive=snapshot.target_state > 0,
            task_phase=snapshot.task_phase,
            has_core=snapshot.core_state == 1,
        )
        return snapshot, privileged

    def _make_observation(
        self,
        screen_buffer,
        snapshot: ProtocolSnapshot,
        game: Any,
        previous_action: MacroAction = MacroAction.IDLE,
    ) -> ActorObservation:
        frame = np.asarray(screen_buffer)
        if frame.ndim != 2:
            raise RuntimeError(f"Actor screen must be grayscale, got shape {frame.shape}")
        resized = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA).astype(
            np.uint8,
            copy=False,
        )
        if self._variant is None:
            raise RuntimeError("Task variant is not initialized")
        return ActorObservation(
            frame=resized,
            health=float(game.get_game_variable(vzd.GameVariable.HEALTH)),
            ammo=float(game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)),
            attack_ready=bool(game.get_game_variable(vzd.GameVariable.ATTACK_READY)),
            has_core=snapshot.core_state == 1,
            home_score=snapshot.home_score,
            away_score=snapshot.away_score,
            remaining_tics=max(0, self._variant.timeout_tics - snapshot.engine_tic),
            previous_action=previous_action,
        )

    def _load_scenario_hash(self) -> str:
        manifest_path = self._config_path.parent / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Scenario manifest does not exist: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        wad_hash = payload.get("wad_sha256")
        if not isinstance(wad_hash, str) or not wad_hash:
            raise ValueError(f"Scenario manifest has no wad_sha256: {manifest_path}")
        return wad_hash

    def _require_game(self) -> Any:
        if self._game is None:
            raise RuntimeError("Environment must be reset before step")
        return self._game
