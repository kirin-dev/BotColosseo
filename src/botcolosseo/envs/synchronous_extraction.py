from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_worker import (
    DuelWorkerSettings,
    WorkerRole,
    spawn_duel_worker,
)
from botcolosseo.envs.extraction_protocol import (
    ExtractionEventDecoder,
    ExtractionProtocolSnapshot,
)
from botcolosseo.envs.extraction_rules import RAID_TIMEOUT_TIC
from botcolosseo.envs.extraction_types import (
    ExtractionActorObservation,
    ExtractionObservations,
    ExtractionPrivilegedState,
    ExtractionResetInfo,
    ExtractionStep,
    normalized_extraction_progress,
    observation_health,
)
from botcolosseo.envs.synchronous_duel import allocate_loopback_port


class SynchronousExtractionEnv:
    def __init__(
        self,
        *,
        config_path: Path,
        seed: int,
        frame_skip: int = 4,
        max_decisions: int = 700,
        worker_timeout: float = 15.0,
        client_factory: Callable[[DuelWorkerSettings], Any] = spawn_duel_worker,
        port_allocator: Callable[[], int] = allocate_loopback_port,
        layout_variant: int | None = None,
    ) -> None:
        if frame_skip <= 0 or max_decisions <= 0 or worker_timeout <= 0:
            raise ValueError("Extraction timing values must be positive")
        self._config_path = config_path.expanduser().resolve()
        self._seed = seed
        self._frame_skip = frame_skip
        self._max_decisions = max_decisions
        self._worker_timeout = worker_timeout
        self._client_factory = client_factory
        self._port_allocator = port_allocator
        self._layout_variant = layout_variant
        self._decoder = ExtractionEventDecoder()
        self._host: Any | None = None
        self._opponent: Any | None = None
        self._port = -1
        self._episode_id = -1
        self._decision_index = 0
        self._last_host: ExtractionActorObservation | None = None
        self._last_opponent: ExtractionActorObservation | None = None
        self._last_snapshot: ExtractionProtocolSnapshot | None = None
        self._privileged: ExtractionPrivilegedState | None = None
        self._scenario_hash = self._load_scenario_hash()

    def reset(self) -> tuple[ExtractionObservations, ExtractionResetInfo]:
        try:
            if self._host is None or self._opponent is None:
                host_state, opponent_state = self._start_workers()
            else:
                host_id = self._host.submit("reset", None)
                opponent_id = self._opponent.submit("reset", None)
                host_state = self._host.receive(host_id)
                opponent_state = self._opponent.receive(opponent_id)
            host_state, opponent_state = self._warmup(host_state, opponent_state)
            snapshot, engine_tic = self._validate_pair(host_state, opponent_state)
            self._episode_id += 1
            self._decision_index = 0
            self._decoder.reset(snapshot)
            observations = self._make_observations(
                host_state,
                opponent_state,
                snapshot,
                MacroAction.IDLE,
                MacroAction.IDLE,
            )
            self._last_snapshot = snapshot
            self._update_privileged(host_state, opponent_state, snapshot)
            return observations, ExtractionResetInfo(
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
    ) -> ExtractionStep:
        host = self._require_client(self._host)
        opponent = self._require_client(self._opponent)
        host_macro = MacroAction(host_action)
        opponent_macro = MacroAction(opponent_action)
        try:
            host_state: dict[str, object] | None = None
            opponent_state: dict[str, object] | None = None
            for tic_index in range(self._frame_skip):
                update_state = tic_index == self._frame_skip - 1
                host_id = host.submit(
                    "step",
                    {"action": int(host_macro), "update_state": update_state},
                )
                opponent_id = opponent.submit(
                    "step",
                    {"action": int(opponent_macro), "update_state": update_state},
                )
                host_state = host.receive(host_id)
                opponent_state = opponent.receive(opponent_id)
                self._validate_engine_tic(host_state, opponent_state)
                if bool(host_state["finished"] or opponent_state["finished"]):
                    break
            if host_state is None or opponent_state is None:
                raise RuntimeError("Extraction step advanced no engine tics")
            snapshot, engine_tic = self._validate_pair(host_state, opponent_state)
            self._decision_index += 1
            events = self._decoder.decode(
                snapshot,
                episode_id=self._episode_id,
                decision_index=self._decision_index,
            )
            observations = self._make_observations(
                host_state,
                opponent_state,
                snapshot,
                host_macro,
                opponent_macro,
            )
            rewards = self._rewards(snapshot)
            self._last_snapshot = snapshot
            self._update_privileged(host_state, opponent_state, snapshot)
            terminated = snapshot.round_state == 2
            engine_finished = bool(
                host_state["finished"] or opponent_state["finished"]
            )
            truncated = not terminated and (
                engine_finished or self._decision_index >= self._max_decisions
            )
            return ExtractionStep(
                host=observations.host,
                opponent=observations.opponent,
                host_reward=rewards[0],
                opponent_reward=rewards[1],
                terminated=terminated,
                truncated=truncated,
                winner=snapshot.winner,
                events=events,
                decision_index=self._decision_index,
                engine_tic=engine_tic,
                peer_tic_lag=abs(
                    int(host_state["protocol_values"][1])
                    - int(opponent_state["protocol_values"][1])
                ),
            )
        except BaseException:
            self.close()
            raise

    def privileged_state(self) -> ExtractionPrivilegedState:
        if self._privileged is None:
            raise RuntimeError("Extraction environment must be reset first")
        return self._privileged

    def protocol_snapshot(self) -> ExtractionProtocolSnapshot:
        if self._last_snapshot is None:
            raise RuntimeError("Extraction environment must be reset first")
        return self._last_snapshot

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
            "map_name": "MAP01",
            "protocol_user_indices": tuple(range(1, 54)),
            "force_respawn": False,
            "time_limit_minutes": 1.5,
            "deathmatch": False,
            "layout_variant": self._layout_variant,
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
        return self._host.receive(host_id), self._opponent.receive(opponent_id)

    def _warmup(
        self,
        host_state: dict[str, object],
        opponent_state: dict[str, object],
        *,
        max_tics: int = 70,
    ) -> tuple[dict[str, object], dict[str, object]]:
        host = self._require_client(self._host)
        opponent = self._require_client(self._opponent)
        for _ in range(max_tics + 1):
            snapshot, _ = self._validate_pair(host_state, opponent_state)
            if (
                snapshot.round_state == 1
                and host_state["frame"] is not None
                and opponent_state["frame"] is not None
                and not bool(host_state["finished"] or opponent_state["finished"])
            ):
                return host_state, opponent_state
            host_id = self._submit_idle(host)
            opponent_id = self._submit_idle(opponent)
            host_state = host.receive(host_id)
            opponent_state = opponent.receive(opponent_id)
        snapshot, _ = self._validate_pair(host_state, opponent_state)
        raise RuntimeError(
            "Extraction players did not become ready within 70 tics: "
            f"protocol={snapshot.protocol_version}, "
            f"round={snapshot.round_state}, "
            f"life={(snapshot.host_life_state, snapshot.opponent_life_state)}, "
            f"health={(host_state['health'], opponent_state['health'])}, "
            f"frames={(host_state['frame'] is not None, opponent_state['frame'] is not None)}, "
            f"finished={(host_state['finished'], opponent_state['finished'])}"
        )

    @staticmethod
    def _submit_idle(client: Any) -> int:
        return client.submit(
            "step",
            {"action": int(MacroAction.IDLE), "update_state": True},
        )

    def _validate_pair(
        self,
        host_state: dict[str, object],
        opponent_state: dict[str, object],
    ) -> tuple[ExtractionProtocolSnapshot, int]:
        host_tic = self._validate_engine_tic(host_state, opponent_state)
        host_snapshot = ExtractionProtocolSnapshot.from_values(
            host_state["protocol_values"]
        )
        ExtractionProtocolSnapshot.from_values(opponent_state["protocol_values"])
        return host_snapshot, host_tic

    @staticmethod
    def _validate_engine_tic(
        host_state: dict[str, object],
        opponent_state: dict[str, object],
    ) -> int:
        host_tic = int(host_state["protocol_values"][1])
        opponent_tic = int(opponent_state["protocol_values"][1])
        if abs(host_tic - opponent_tic) > 2:
            raise RuntimeError(
                "Extraction protocol tic mismatch exceeds replication tolerance: "
                f"host={host_tic}, opponent={opponent_tic}"
            )
        return host_tic

    def _make_observations(
        self,
        host_state: dict[str, object],
        opponent_state: dict[str, object],
        snapshot: ExtractionProtocolSnapshot,
        host_action: MacroAction,
        opponent_action: MacroAction,
    ) -> ExtractionObservations:
        host = self._make_observation(
            host_state,
            snapshot,
            "host",
            host_action,
            self._last_host,
        )
        opponent = self._make_observation(
            opponent_state,
            snapshot,
            "opponent",
            opponent_action,
            self._last_opponent,
        )
        self._last_host = host
        self._last_opponent = opponent
        return ExtractionObservations(host=host, opponent=opponent)

    @staticmethod
    def _make_observation(
        state: dict[str, object],
        snapshot: ExtractionProtocolSnapshot,
        side: str,
        previous_action: MacroAction,
        fallback: ExtractionActorObservation | None,
    ) -> ExtractionActorObservation:
        raw_frame = state["frame"]
        if raw_frame is None:
            if fallback is None:
                raise RuntimeError("Extraction worker returned no initial frame")
            frame = fallback.frame
        else:
            array = np.asarray(raw_frame)
            if array.ndim != 2:
                raise RuntimeError(
                    f"Extraction frame must be grayscale, got {array.shape}"
                )
            frame = cv2.resize(
                array,
                (84, 84),
                interpolation=cv2.INTER_AREA,
            ).astype(np.uint8, copy=False)
        public = snapshot.public_state(side)
        return ExtractionActorObservation(
            frame=frame,
            health=observation_health(public.life_state, float(state["health"])),
            ammo=max(0.0, min(float(state["ammo"]), 40.0)),
            carried_value=public.carried_value,
            free_slots=public.free_slots,
            minimum_slot_value=public.minimum_slot_value,
            banked_value=public.banked_value,
            extraction_open=public.extraction_open,
            extraction_progress=normalized_extraction_progress(
                public.extraction_progress_tics
            ),
            remaining_time=max(0.0, (RAID_TIMEOUT_TIC - snapshot.engine_tic) / 35.0),
            previous_action=int(previous_action),
        )

    def _rewards(
        self,
        snapshot: ExtractionProtocolSnapshot,
    ) -> tuple[float, float]:
        previous = self._last_snapshot
        if previous is None:
            return 0.0, 0.0
        host = (snapshot.host_banked - previous.host_banked) / 50.0
        opponent = (snapshot.opponent_banked - previous.opponent_banked) / 50.0
        if snapshot.round_state == 2:
            if snapshot.winner == 1:
                host += 1.0
                opponent -= 1.0
            elif snapshot.winner == 2:
                host -= 1.0
                opponent += 1.0
        return host, opponent

    def _update_privileged(
        self,
        host_state: dict[str, object],
        opponent_state: dict[str, object],
        snapshot: ExtractionProtocolSnapshot,
    ) -> None:
        host_public = snapshot.public_state("host")
        opponent_public = snapshot.public_state("opponent")
        self._privileged = ExtractionPrivilegedState(
            host_x=float(host_state["player_x"]),
            host_y=float(host_state["player_y"]),
            host_angle=float(host_state["player_angle"]),
            opponent_x=float(opponent_state["player_x"]),
            opponent_y=float(opponent_state["player_y"]),
            opponent_angle=float(opponent_state["player_angle"]),
            host_health=observation_health(
                host_public.life_state,
                float(host_state["health"]),
            ),
            opponent_health=observation_health(
                opponent_public.life_state,
                float(opponent_state["health"]),
            ),
            host_slots=host_public.slots,
            opponent_slots=opponent_public.slots,
            host_banked=snapshot.host_banked,
            opponent_banked=snapshot.opponent_banked,
            cache_owner=snapshot.cache_owner,
            cache_slots=(
                snapshot.cache_slot_0,
                snapshot.cache_slot_1,
                snapshot.cache_slot_2,
            ),
            cache_x=snapshot.cache_x,
            cache_y=snapshot.cache_y,
            world_loot_mask=snapshot.world_loot_mask,
            round_state=snapshot.round_state,
            winner=snapshot.winner,
            engine_tic=snapshot.engine_tic,
        )

    def _load_scenario_hash(self) -> str:
        manifest_path = self._config_path.parent / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        wad_hash = payload.get("wad_sha256")
        if not isinstance(wad_hash, str) or not wad_hash:
            raise ValueError(
                f"Extraction manifest has no WAD hash: {manifest_path}"
            )
        return wad_hash

    @staticmethod
    def _require_client(client: Any | None) -> Any:
        if client is None:
            raise RuntimeError("Extraction environment must be reset before step")
        return client
