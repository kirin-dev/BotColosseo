from __future__ import annotations

import json
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_protocol import (
    DuelEvent,
    DuelEventDecoder,
    DuelEventType,
    DuelProtocolSnapshot,
)
from botcolosseo.envs.duel_rewards import DuelRewardLedger, load_reward_config
from botcolosseo.envs.duel_types import (
    DuelActorObservation,
    DuelPrivilegedState,
    DuelStep,
)
from botcolosseo.envs.duel_worker import (
    DuelWorkerSettings,
    WorkerRole,
    spawn_duel_worker,
)
from botcolosseo.scenarios.regions import RegionGraph


@dataclass(frozen=True)
class DuelObservations:
    host: DuelActorObservation
    opponent: DuelActorObservation


@dataclass(frozen=True)
class DuelResetInfo:
    seed: int
    port: int
    episode_id: int
    engine_tic: int
    protocol_version: int
    scenario_hash: str


def allocate_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class SynchronousDuelEnv:
    def __init__(
        self,
        *,
        config_path: Path,
        region_graph: RegionGraph,
        seed: int,
        frame_skip: int = 4,
        max_decisions: int = 525,
        worker_timeout: float = 15.0,
        client_factory: Callable[[DuelWorkerSettings], Any] = spawn_duel_worker,
        port_allocator: Callable[[], int] = allocate_loopback_port,
    ) -> None:
        if frame_skip <= 0 or max_decisions <= 0 or worker_timeout <= 0:
            raise ValueError("Duel timing values must be positive")
        self._config_path = config_path.expanduser().resolve()
        self._graph = region_graph
        self._seed = seed
        self._frame_skip = frame_skip
        self._max_decisions = max_decisions
        self._worker_timeout = worker_timeout
        self._client_factory = client_factory
        self._port_allocator = port_allocator
        root = self._config_path.parents[3]
        self._ledger = DuelRewardLedger(
            load_reward_config(root / "configs/m2/reward.yaml")
        )
        self._decoder = DuelEventDecoder()
        self._host: Any | None = None
        self._opponent: Any | None = None
        self._port = -1
        self._episode_id = -1
        self._decision_index = 0
        self._last_host: DuelActorObservation | None = None
        self._last_opponent: DuelActorObservation | None = None
        self._last_host_state: dict[str, object] | None = None
        self._last_opponent_state: dict[str, object] | None = None
        self._privileged: DuelPrivilegedState | None = None
        self._previous_hitcounts = {"host": 0, "opponent": 0}
        self._shaping_scale = 1.0
        self._scenario_hash = self._load_scenario_hash()

    def set_shaping_scale(self, scale: float) -> None:
        if not 0.0 <= scale <= 1.0:
            raise ValueError("shaping scale must be in [0, 1]")
        self._shaping_scale = scale

    def reset(self) -> tuple[DuelObservations, DuelResetInfo]:
        try:
            if self._host is None or self._opponent is None:
                host_state, opponent_state = self._start_workers()
            else:
                host_id = self._host.submit("reset", None)
                opponent_id = self._opponent.submit("reset", None)
                host_state = self._host.receive(host_id)
                opponent_state = self._opponent.receive(opponent_id)
            host_state, opponent_state = self._warmup_players(
                host_state, opponent_state
            )
            snapshot, engine_tic = self._validate_pair(host_state, opponent_state)
            self._episode_id += 1
            self._decision_index = 0
            self._ledger.reset()
            self._decoder.reset(snapshot)
            self._previous_hitcounts = {
                "host": int(host_state["hitcount"]),
                "opponent": int(opponent_state["hitcount"]),
            }
            observations = self._make_observations(
                host_state,
                opponent_state,
                snapshot,
                MacroAction.IDLE,
                MacroAction.IDLE,
            )
            self._last_host_state = host_state
            self._last_opponent_state = opponent_state
            self._update_privileged(host_state, opponent_state, snapshot, engine_tic)
            return observations, DuelResetInfo(
                seed=self._seed,
                port=self._port,
                episode_id=self._episode_id,
                engine_tic=engine_tic,
                protocol_version=snapshot.protocol_version,
                scenario_hash=self._scenario_hash,
            )
        except BaseException:
            self.close()
            raise

    def step(
        self,
        host_action: MacroAction | int,
        opponent_action: MacroAction | int,
    ) -> DuelStep:
        host = self._require_client(self._host)
        opponent = self._require_client(self._opponent)
        host_macro = MacroAction(host_action)
        opponent_macro = MacroAction(opponent_action)
        try:
            pre_action_tics = self._ensure_players_alive()
            if self._last_host_state is None:
                raise RuntimeError("Duel environment has no pre-action host state")
            action_start_tic = int(self._last_host_state["protocol_values"][1])
            if bool(
                self._last_host_state["finished"]
                or self._last_opponent_state["finished"]
            ):
                return self._finalize_step(
                    self._last_host_state,
                    self._last_opponent_state,
                    MacroAction.IDLE,
                    MacroAction.IDLE,
                    pre_action_tics=pre_action_tics,
                    action_start_tic=action_start_tic,
                )
            host_state: dict[str, object] | None = None
            opponent_state: dict[str, object] | None = None
            for tic_index in range(self._frame_skip):
                update_state = tic_index == self._frame_skip - 1
                host_id = host.submit(
                    "step", {"action": int(host_macro), "update_state": update_state}
                )
                opponent_id = opponent.submit(
                    "step",
                    {"action": int(opponent_macro), "update_state": update_state},
                )
                host_state = host.receive(host_id)
                opponent_state = opponent.receive(opponent_id)
                self._validate_engine_tic(host_state, opponent_state)
            if host_state is None or opponent_state is None:
                raise RuntimeError("Duel step advanced no engine tics")
            return self._finalize_step(
                host_state,
                opponent_state,
                host_macro,
                opponent_macro,
                pre_action_tics=pre_action_tics,
                action_start_tic=action_start_tic,
            )
        except BaseException:
            self.close()
            raise

    def _finalize_step(
        self,
        host_state: dict[str, object],
        opponent_state: dict[str, object],
        host_action: MacroAction,
        opponent_action: MacroAction,
        *,
        pre_action_tics: int,
        action_start_tic: int,
    ) -> DuelStep:
        snapshot, engine_tic = self._validate_pair(host_state, opponent_state)
        self._decision_index += 1
        events = self._with_native_hit_events(
            self._decoder.decode(
                snapshot,
                episode_id=self._episode_id,
                decision_index=self._decision_index,
            ),
            host_state,
            opponent_state,
            engine_tic,
        )
        rewards = self._ledger.apply(events, shaping_scale=self._shaping_scale)
        observations = self._make_observations(
            host_state,
            opponent_state,
            snapshot,
            host_action,
            opponent_action,
        )
        self._update_privileged(host_state, opponent_state, snapshot, engine_tic)
        self._last_host_state = host_state
        self._last_opponent_state = opponent_state
        terminated = snapshot.winner > 0 or snapshot.round_state == 2
        engine_finished = bool(host_state["finished"] or opponent_state["finished"])
        truncated = not terminated and (
            engine_finished or self._decision_index >= self._max_decisions
        )
        return DuelStep(
            host=observations.host,
            opponent=observations.opponent,
            host_reward=rewards.host,
            opponent_reward=rewards.opponent,
            terminated=terminated,
            truncated=truncated,
            events=events,
            decision_index=self._decision_index,
            engine_tic=engine_tic,
            peer_tic_lag=abs(
                int(host_state["protocol_values"][1])
                - int(opponent_state["protocol_values"][1])
            ),
            pre_action_tics=pre_action_tics,
            action_tics=engine_tic - action_start_tic,
        )

    def teacher_state(self) -> DuelPrivilegedState:
        if self._privileged is None:
            raise RuntimeError("Duel environment must be reset before Teacher access")
        return self._privileged

    def close(self) -> None:
        opponent, host = self._opponent, self._host
        self._opponent = None
        self._host = None
        for client in (opponent, host):
            if client is not None:
                client.close()

    def _start_workers(self) -> tuple[dict[str, object], dict[str, object]]:
        self._port = self._port_allocator()
        common = {
            "config_path": self._config_path,
            "seed": self._seed,
            "port": self._port,
            "timeout": self._worker_timeout,
        }
        self._host = self._client_factory(
            DuelWorkerSettings(role=WorkerRole.HOST, **common)
        )
        self._opponent = self._client_factory(
            DuelWorkerSettings(role=WorkerRole.OPPONENT, **common)
        )
        host_id = self._host.submit("init", None)
        time.sleep(0.1)
        opponent_id = self._opponent.submit("init", None)
        host_state = self._host.receive(host_id)
        opponent_state = self._opponent.receive(opponent_id)
        return host_state, opponent_state

    def _ensure_players_alive(self) -> int:
        if self._last_host_state is None or self._last_opponent_state is None:
            return 0
        if not (
            bool(self._last_host_state["dead"])
            or bool(self._last_opponent_state["dead"])
        ):
            return 0
        previous_tic = int(self._last_host_state["protocol_values"][1])
        host = self._require_client(self._host)
        opponent = self._require_client(self._opponent)
        host_state, opponent_state = self._respawn_players(
            host,
            opponent,
            self._last_host_state,
            self._last_opponent_state,
            max_tics=70,
        )
        snapshot, engine_tic = self._validate_pair(host_state, opponent_state)
        self._make_observations(
            host_state,
            opponent_state,
            snapshot,
            MacroAction.IDLE,
            MacroAction.IDLE,
        )
        self._update_privileged(host_state, opponent_state, snapshot, engine_tic)
        self._last_host_state = host_state
        self._last_opponent_state = opponent_state
        return engine_tic - previous_tic

    def _warmup_players(
        self,
        host_state: dict[str, object],
        opponent_state: dict[str, object],
        *,
        max_tics: int = 70,
    ) -> tuple[dict[str, object], dict[str, object]]:
        host = self._require_client(self._host)
        opponent = self._require_client(self._opponent)
        for _ in range(max_tics + 1):
            protocol_tic = int(host_state["protocol_values"][1])
            if (
                self._valid_player_state(host_state)
                and self._valid_player_state(opponent_state)
                and protocol_tic >= 10
            ):
                return host_state, opponent_state
            if bool(host_state["dead"]) or bool(opponent_state["dead"]):
                return self._respawn_players(
                    host,
                    opponent,
                    host_state,
                    opponent_state,
                    max_tics=max_tics,
                )
            host_id = self._submit_idle(host)
            opponent_id = self._submit_idle(opponent)
            host_state = host.receive(host_id)
            opponent_state = opponent.receive(opponent_id)
            self._validate_engine_tic(host_state, opponent_state)
        raise RuntimeError("Duel players did not enter a valid state within 70 tics")

    @staticmethod
    def _valid_player_state(state: dict[str, object]) -> bool:
        return (
            state["frame"] is not None
            and float(state["health"]) >= 0.0
            and not bool(state["dead"])
            and not bool(state["finished"])
        )

    @staticmethod
    def _submit_idle(client: Any) -> int:
        return client.submit(
            "step", {"action": int(MacroAction.IDLE), "update_state": True}
        )

    def _respawn_players(
        self,
        host: Any,
        opponent: Any,
        host_state: dict[str, object],
        opponent_state: dict[str, object],
        *,
        max_tics: int,
    ) -> tuple[dict[str, object], dict[str, object]]:
        for _ in range(max_tics + 1):
            if bool(host_state["finished"] or opponent_state["finished"]):
                return host_state, opponent_state
            if self._valid_player_state(host_state) and self._valid_player_state(
                opponent_state
            ):
                return host_state, opponent_state
            host_id = self._submit_idle(host)
            opponent_id = self._submit_idle(opponent)
            host_state = host.receive(host_id)
            opponent_state = opponent.receive(opponent_id)
            self._validate_engine_tic(host_state, opponent_state)
        raise RuntimeError("Duel respawn did not complete within the warm-up limit")

    def _validate_pair(
        self, host_state: dict[str, object], opponent_state: dict[str, object]
    ) -> tuple[DuelProtocolSnapshot, int]:
        host_tic = self._validate_engine_tic(host_state, opponent_state)
        host_snapshot = DuelProtocolSnapshot.from_values(host_state["protocol_values"])
        DuelProtocolSnapshot.from_values(opponent_state["protocol_values"])
        return host_snapshot, host_tic

    @staticmethod
    def _validate_engine_tic(
        host_state: dict[str, object], opponent_state: dict[str, object]
    ) -> int:
        host_tic = int(host_state["protocol_values"][1])
        opponent_tic = int(opponent_state["protocol_values"][1])
        if abs(host_tic - opponent_tic) > 2:
            raise RuntimeError(
                "Duel protocol tic mismatch exceeds replication tolerance: "
                f"host={host_tic}, opponent={opponent_tic}"
            )
        return host_tic

    def _make_observations(
        self,
        host_state: dict[str, object],
        opponent_state: dict[str, object],
        snapshot: DuelProtocolSnapshot,
        host_action: MacroAction,
        opponent_action: MacroAction,
    ) -> DuelObservations:
        host = self._make_observation(
            host_state,
            own_score=snapshot.host_score,
            opponent_score=snapshot.opponent_score,
            has_core=snapshot.carrier == 1,
            previous_action=host_action,
            fallback=self._last_host,
        )
        opponent = self._make_observation(
            opponent_state,
            own_score=snapshot.opponent_score,
            opponent_score=snapshot.host_score,
            has_core=snapshot.carrier == 2,
            previous_action=opponent_action,
            fallback=self._last_opponent,
        )
        self._last_host = host
        self._last_opponent = opponent
        return DuelObservations(host, opponent)

    def _with_native_hit_events(
        self,
        events: tuple[DuelEvent, ...],
        host_state: dict[str, object],
        opponent_state: dict[str, object],
        engine_tic: int,
    ) -> tuple[DuelEvent, ...]:
        augmented = list(events)
        existing = {
            event.side
            for event in events
            if event.type is DuelEventType.VALID_HIT
        }
        for side, state in (("host", host_state), ("opponent", opponent_state)):
            current = int(state["hitcount"])
            previous = self._previous_hitcounts[side]
            if current > previous and side not in existing:
                augmented.append(
                    DuelEvent(
                        type=DuelEventType.VALID_HIT,
                        side=side,
                        episode_id=self._episode_id,
                        decision_index=self._decision_index,
                        engine_tic=engine_tic,
                    )
                )
            self._previous_hitcounts[side] = current
        return tuple(augmented)

    @staticmethod
    def _make_observation(
        state: dict[str, object],
        *,
        own_score: int,
        opponent_score: int,
        has_core: bool,
        previous_action: MacroAction,
        fallback: DuelActorObservation | None,
    ) -> DuelActorObservation:
        raw_frame = state["frame"]
        if raw_frame is None:
            if fallback is None:
                raise RuntimeError("Duel worker returned no initial frame")
            frame = fallback.frame
        else:
            array = np.asarray(raw_frame)
            if array.ndim != 2:
                raise RuntimeError(f"Duel frame must be grayscale, got {array.shape}")
            frame = cv2.resize(array, (84, 84), interpolation=cv2.INTER_AREA).astype(
                np.uint8, copy=False
            )
        return DuelActorObservation(
            frame=frame,
            health=0.0 if bool(state["dead"]) else float(state["health"]),
            armor=float(state["armor"]),
            ammo=float(state["ammo"]),
            own_score=own_score,
            opponent_score=opponent_score,
            has_core=has_core,
            previous_action=int(previous_action),
        )

    def _update_privileged(
        self,
        host_state: dict[str, object],
        opponent_state: dict[str, object],
        snapshot: DuelProtocolSnapshot,
        engine_tic: int,
    ) -> None:
        host_region = self._graph.region_at(
            float(host_state["player_x"]), float(host_state["player_y"])
        )
        opponent_region = self._graph.region_at(
            float(opponent_state["player_x"]), float(opponent_state["player_y"])
        )
        self._privileged = DuelPrivilegedState(
            host_x=float(host_state["player_x"]),
            host_y=float(host_state["player_y"]),
            host_angle=float(host_state["player_angle"]),
            host_region=None if host_region is None else host_region.name,
            opponent_x=float(opponent_state["player_x"]),
            opponent_y=float(opponent_state["player_y"]),
            opponent_angle=float(opponent_state["player_angle"]),
            opponent_region=None if opponent_region is None else opponent_region.name,
            core_x=float(snapshot.core_x),
            core_y=float(snapshot.core_y),
            carrier=snapshot.carrier,
            host_health=(0.0 if bool(host_state["dead"]) else float(host_state["health"])),
            opponent_health=(
                0.0
                if bool(opponent_state["dead"])
                else float(opponent_state["health"])
            ),
            host_score=snapshot.host_score,
            opponent_score=snapshot.opponent_score,
            round_state=snapshot.round_state,
            engine_tic=engine_tic,
        )

    def _load_scenario_hash(self) -> str:
        manifest_path = self._config_path.parent / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        wad_hash = payload.get("wad_sha256")
        if not isinstance(wad_hash, str) or not wad_hash:
            raise ValueError(f"Scenario manifest has no WAD hash: {manifest_path}")
        return wad_hash

    @staticmethod
    def _require_client(client: Any | None) -> Any:
        if client is None:
            raise RuntimeError("Duel environment must be reset before step")
        return client
