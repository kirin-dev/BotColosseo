from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from botcolosseo.agents.extraction_model import load_extraction_policy
from botcolosseo.agents.extraction_teachers import StyledExtractionTeacher
from botcolosseo.data.extraction_demonstrations import ExtractionCase
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_protocol import (
    ExtractionEvent,
    ExtractionEventType,
    ExtractionProtocolSnapshot,
)
from botcolosseo.envs.extraction_types import (
    ExtractionActorObservation,
    ExtractionPrivilegedState,
)
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv
from botcolosseo.training.extraction_bc import extraction_observation_tensors


@dataclass(frozen=True)
class RecordedExtractionShowcase:
    style: str
    case: ExtractionCase
    frames: tuple[NDArray[np.uint8], ...]
    decisions: int
    extracted_value: int
    extracted: bool
    died: bool
    events: dict[str, int]
    scenario_hash: str
    test_cases_accessed: bool = False

    def record(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("frames")
        return payload


def _event_label(
    events: tuple[ExtractionEvent, ...],
    *,
    learner_side: str,
) -> str | None:
    opponent_side = "opponent" if learner_side == "host" else "host"
    labels: list[str] = []
    for event in events:
        if event.type is ExtractionEventType.VALID_HIT:
            labels.append(
                "HIT CONFIRMED  -20 HP"
                if event.side == learner_side
                else "TAKING FIRE  -20 HP"
            )
        elif event.type is ExtractionEventType.DEATH:
            labels.append(
                "ENEMY DOWN  ->  CORPSE CACHE"
                if event.side == opponent_side
                else "ELIMINATED  ->  LOOT LOST"
            )
        elif event.type is ExtractionEventType.CACHE_CREATED:
            labels.append("CORPSE CACHE CREATED")
        elif event.type is ExtractionEventType.CACHE_LOOTED:
            labels.append(f"CACHE LOOTED  +{event.value}")
        elif event.type is ExtractionEventType.LOOT_PICKUP:
            labels.append(f"LOOT PICKUP  +{event.value}")
        elif event.type is ExtractionEventType.LOOT_DROP:
            labels.append(f"BACKPACK UPGRADE  DROP {event.value}")
        elif event.type is ExtractionEventType.EXTRACTION_STARTED:
            labels.append("EXTRACTION STARTED  HOLD 3s")
        elif event.type is ExtractionEventType.EXTRACTION_INTERRUPTED:
            labels.append("EXTRACTION INTERRUPTED")
        elif event.type is ExtractionEventType.EXTRACTED:
            labels.append("VALUE BANKED  EXTRACTION COMPLETE")
    return " | ".join(labels) if labels else None


def _draw_bar(
    canvas: NDArray[np.uint8],
    *,
    origin: tuple[int, int],
    width: int,
    value: float,
    maximum: float,
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    cv2.rectangle(canvas, (x, y), (x + width, y + 10), (70, 70, 70), -1)
    filled = round(width * max(0.0, min(value / maximum, 1.0)))
    cv2.rectangle(canvas, (x, y), (x + filled, y + 10), color, -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + 10), (210, 210, 210), 1)


def compose_extraction_showcase_frame(
    observation: ExtractionActorObservation,
    *,
    privileged: ExtractionPrivilegedState,
    protocol: ExtractionProtocolSnapshot,
    learner_side: str,
    style: str,
    action: MacroAction,
    event_label: str,
) -> NDArray[np.uint8]:
    view = cv2.resize(observation.frame, (480, 270), interpolation=cv2.INTER_NEAREST)
    view = cv2.cvtColor(view, cv2.COLOR_GRAY2RGB)
    canvas = np.full((360, 640, 3), 14, dtype=np.uint8)
    canvas[50:320, :480] = view
    self_health, enemy_health = (
        (privileged.host_health, privileged.opponent_health)
        if learner_side == "host"
        else (privileged.opponent_health, privileged.host_health)
    )
    own = protocol.public_state(learner_side)
    enemy_side = "opponent" if learner_side == "host" else "host"
    enemy = protocol.public_state(enemy_side)

    cv2.putText(
        canvas,
        "CRYSTAL RUN: EXTRACTION",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{style.upper()} BOT  |  FIRST-PERSON POLICY",
        (10, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (80, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "VIEWER TELEMETRY",
        (500, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"SELF HP {self_health:.0f}",
        (490, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    _draw_bar(
        canvas,
        origin=(490, 53),
        width=135,
        value=self_health,
        maximum=100,
        color=(70, 220, 100),
    )
    cv2.putText(
        canvas,
        f"ENEMY HP {enemy_health:.0f}",
        (490, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    _draw_bar(
        canvas,
        origin=(490, 89),
        width=135,
        value=enemy_health,
        maximum=100,
        color=(240, 90, 80),
    )
    cv2.putText(
        canvas,
        f"AMMO  {observation.ammo:.0f} / 40",
        (490, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "BACKPACK",
        (490, 148),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    for index, value in enumerate(own.slots):
        x = 490 + index * 46
        cv2.rectangle(canvas, (x, 157), (x + 38, 181), (110, 110, 110), 1)
        cv2.putText(
            canvas,
            str(value) if value else "-",
            (x + 10, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (90, 220, 255) if value else (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        f"CARRIED  {own.carried_value:3d}",
        (490, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"BANKED   {own.banked_value:3d}",
        (490, 226),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (100, 230, 130),
        1,
        cv2.LINE_AA,
    )
    cache_total = protocol.cache_slot_0 + protocol.cache_slot_1 + protocol.cache_slot_2
    cv2.putText(
        canvas,
        f"ENEMY CARRY {enemy.carried_value:3d}",
        (490, 247),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"CORPSE CACHE {cache_total:3d}",
        (490, 266),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )
    extraction_state = "OPEN" if observation.extraction_open else "LOCKED (30s)"
    cv2.putText(
        canvas,
        f"EXTRACT {extraction_state}",
        (490, 289),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (90, 220, 255) if observation.extraction_open else (160, 160, 160),
        1,
        cv2.LINE_AA,
    )
    _draw_bar(
        canvas,
        origin=(490, 298),
        width=135,
        value=observation.extraction_progress,
        maximum=1.0,
        color=(90, 220, 255),
    )
    cv2.putText(
        canvas,
        f"ACTION {action.name}",
        (490, 319),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.31,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        event_label,
        (10, 341),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (80, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "SEARCH  >  FIGHT OR EVADE  >  LOOT  >  EXTRACT",
        (298, 341),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.31,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return canvas


@torch.no_grad()
def record_extraction_showcase(
    *,
    root: Path,
    checkpoint: Path,
    style: str,
    case: ExtractionCase,
    device: torch.device,
    frame_stride: int = 2,
    max_decisions: int = 700,
) -> RecordedExtractionShowcase:
    if case.split != "validation":
        raise ValueError("Extraction showcase must use validation cases")
    if frame_stride <= 0:
        raise ValueError("Extraction showcase frame stride must be positive")
    scenario_hash = json.loads(
        (
            root / "assets/scenarios/crystal_run_extraction/manifest.json"
        ).read_text(encoding="utf-8")
    )["wad_sha256"]
    model, _ = load_extraction_policy(
        checkpoint,
        style=style,
        scenario_hash=scenario_hash,
        device=device,
    )
    env = SynchronousExtractionEnv(
        config_path=root
        / "assets/scenarios/crystal_run_extraction/crystal_run_extraction.cfg",
        seed=case.seed,
        max_decisions=max_decisions,
    )
    opponent_side = "opponent" if case.learner_side == "host" else "host"
    opponent = StyledExtractionTeacher(
        side=opponent_side,
        style=case.opponent_style,
    )
    frames: list[NDArray[np.uint8]] = []
    event_counts: Counter[str] = Counter()
    hidden = model.initial_state(1, device=device)
    event_banner = "ENTER RAID  |  SEARCH FOR VALUE"
    banner_ttl = 0
    decisions = 0
    try:
        observations, reset_info = env.reset()
        opponent.reset()
        episode_start = True
        for _ in range(max_decisions):
            decisions += 1
            observation = (
                observations.host
                if case.learner_side == "host"
                else observations.opponent
            )
            output = model(
                *extraction_observation_tensors(
                    observation,
                    episode_start=episode_start,
                    device=device,
                ),
                hidden,
            )
            learner_action = MacroAction(int(output.logits[0, 0].argmax()))
            hidden = output.hidden
            state = env.privileged_state()
            opponent_action = opponent.act(state)
            host_action, away_action = (
                (learner_action, opponent_action)
                if case.learner_side == "host"
                else (opponent_action, learner_action)
            )
            step = env.step(host_action, away_action)
            observations = type(observations)(step.host, step.opponent)
            for event in step.events:
                event_counts[f"{event.side}:{event.type.value}"] += event.count
            label = _event_label(step.events, learner_side=case.learner_side)
            if label:
                event_banner = label
                banner_ttl = 18
            elif banner_ttl > 0:
                banner_ttl -= 1
            else:
                event_banner = "SEARCHING  |  VALUE ONLY COUNTS AFTER EXTRACTION"
            learner_observation = (
                step.host if case.learner_side == "host" else step.opponent
            )
            if decisions % frame_stride == 0 or label or step.terminated or step.truncated:
                frames.append(
                    compose_extraction_showcase_frame(
                        learner_observation,
                        privileged=env.privileged_state(),
                        protocol=env.protocol_snapshot(),
                        learner_side=case.learner_side,
                        style=style,
                        action=learner_action,
                        event_label=event_banner,
                    )
                )
            episode_start = False
            life = env.protocol_snapshot().public_state(case.learner_side).life_state
            if life != 1 or step.terminated or step.truncated:
                break
        final = (
            observations.host
            if case.learner_side == "host"
            else observations.opponent
        )
        life = env.protocol_snapshot().public_state(case.learner_side).life_state
        return RecordedExtractionShowcase(
            style=style,
            case=case,
            frames=tuple(frames),
            decisions=decisions,
            extracted_value=final.banked_value,
            extracted=life == 3,
            died=life == 2,
            events=dict(sorted(event_counts.items())),
            scenario_hash=reset_info.scenario_hash,
        )
    finally:
        env.close()
