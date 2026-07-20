from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from botcolosseo.agents.duel_teachers import DuelTeacher
from botcolosseo.envs.duel_protocol import DuelEventType
from botcolosseo.envs.duel_types import DuelStep
from botcolosseo.envs.synchronous_duel import SynchronousDuelEnv
from botcolosseo.envs.video import write_mp4


class SyncAuditAccumulator:
    def __init__(self, *, target_decisions: int) -> None:
        if target_decisions <= 0:
            raise ValueError("target_decisions must be positive")
        self._target = target_decisions
        self._completed = 0
        self._episodes = 0
        self._last_tic: int | None = None
        self._death_on_previous_decision = False
        self._events: Counter[str] = Counter()
        self._max_peer_tic_lag = 0

    def start_episode(self, *, initial_tic: int) -> None:
        if initial_tic < 0:
            raise ValueError("initial_tic must be nonnegative")
        self._episodes += 1
        self._last_tic = initial_tic
        self._death_on_previous_decision = False

    def record(self, step: DuelStep) -> None:
        if self._last_tic is None:
            raise RuntimeError("start_episode must be called before record")
        if self._completed >= self._target:
            raise RuntimeError("Audit received more decisions than requested")
        tic_delta = step.engine_tic - self._last_tic
        if tic_delta != 4 and not (
            self._death_on_previous_decision and tic_delta >= 4
        ):
            raise RuntimeError(
                f"Duel decision must advance four tics, observed {tic_delta}"
            )
        if not 0 <= step.peer_tic_lag <= 2:
            raise RuntimeError(f"Peer tic lag exceeds tolerance: {step.peer_tic_lag}")
        self._completed += 1
        self._last_tic = step.engine_tic
        self._max_peer_tic_lag = max(self._max_peer_tic_lag, step.peer_tic_lag)
        self._events.update(f"{event.side}:{event.type.value}" for event in step.events)
        self._death_on_previous_decision = any(
            event.type is DuelEventType.DEATH for event in step.events
        )

    def finish(self, *, cleaned_workers: bool) -> dict[str, object]:
        if self._completed != self._target:
            raise RuntimeError(
                f"Audit requires exactly {self._target} decisions, got {self._completed}"
            )
        return {
            "cleaned_workers": cleaned_workers,
            "completed_decisions": self._completed,
            "episode_count": self._episodes,
            "event_counts": dict(sorted(self._events.items())),
            "max_peer_tic_lag": self._max_peer_tic_lag,
            "protocol_errors": 0,
            "tic_mismatches": 0,
            "worker_timeouts": 0,
            "passed": cleaned_workers and self._max_peer_tic_lag <= 2,
        }


def compose_duel_frame(step: DuelStep, *, event_label: str) -> np.ndarray:
    host = cv2.resize(step.host.frame, (168, 168), interpolation=cv2.INTER_NEAREST)
    opponent = cv2.resize(
        step.opponent.frame, (168, 168), interpolation=cv2.INTER_NEAREST
    )
    arena = np.concatenate((host, opponent), axis=1)
    arena = cv2.cvtColor(arena, cv2.COLOR_GRAY2RGB)
    canvas = np.zeros((200, 336, 3), dtype=np.uint8)
    canvas[32:] = arena
    cv2.putText(
        canvas,
        f"HOST {step.host.own_score}-{step.opponent.own_score} OPP  event={event_label}",
        (5, 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (
            f"hp={step.host.health:.0f} core={int(step.host.has_core)}"
            f"     hp={step.opponent.health:.0f} core={int(step.opponent.has_core)}"
        ),
        (5, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


def run_sync_audit(
    env: SynchronousDuelEnv,
    host_teacher: DuelTeacher,
    opponent_teacher: DuelTeacher,
    *,
    decisions: int,
    seed: int,
    video_path: Path,
    video_frame_cap: int = 200,
) -> dict[str, object]:
    if video_frame_cap <= 0:
        raise ValueError("video_frame_cap must be positive")
    before = {child.pid for child in mp.active_children()}
    audit = SyncAuditAccumulator(target_decisions=decisions)
    frames: list[np.ndarray] = []
    started = time.monotonic()
    episode = 0
    scenario_hash = ""
    try:
        _, info = env.reset()
        scenario_hash = info.scenario_hash
        audit.start_episode(initial_tic=info.engine_tic)
        host_teacher.reset(seed=seed)
        opponent_teacher.reset(seed=seed)
        for completed in range(decisions):
            state = env.teacher_state()
            step = env.step(host_teacher.act(state), opponent_teacher.act(state))
            audit.record(step)
            if len(frames) < video_frame_cap:
                label = ",".join(event.type.value for event in step.events) or "none"
                frames.append(compose_duel_frame(step, event_label=label))
            if completed + 1 < decisions and (step.terminated or step.truncated):
                episode += 1
                _, info = env.reset()
                audit.start_episode(initial_tic=info.engine_tic)
                host_teacher.reset(seed=seed + episode)
                opponent_teacher.reset(seed=seed + episode)
    finally:
        env.close()
    cleaned = {child.pid for child in mp.active_children()} <= before
    summary = audit.finish(cleaned_workers=cleaned)
    summary.update(
        {
            "host_teacher": host_teacher.name,
            "opponent_teacher": opponent_teacher.name,
            "scenario_hash": scenario_hash,
            "seed": seed,
            "video_frames": len(frames),
            "wall_seconds": round(time.monotonic() - started, 3),
        }
    )
    rendered_video = write_mp4(frames, video_path, fps=10)
    summary.update(
        {
            "video_bytes": rendered_video.stat().st_size,
            "video_duration_seconds": len(frames) / 10,
            "video_fps": 10,
            "video_path": str(rendered_video),
        }
    )
    return summary


def write_audit_summary(summary: dict[str, Any], output_path: Path) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    return output_path
