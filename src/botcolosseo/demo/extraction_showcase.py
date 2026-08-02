from __future__ import annotations

import json
import math
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
from botcolosseo.evaluation.extraction import DisengagementTracker
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
    attack_decisions: int
    unique_route_cells: int
    aggressive_chains: int
    successful_disengagements: int
    meaningful_extractions: int
    meaningful_loot_regions: int
    backpack_upgrades: int
    upgrade_to_extraction_conversions: int
    scenario_hash: str
    test_cases_accessed: bool = False

    def record(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("frames")
        return payload


@dataclass(frozen=True)
class _ShowcaseFrameInput:
    observation: ExtractionActorObservation
    privileged: ExtractionPrivilegedState
    protocol: ExtractionProtocolSnapshot
    action: MacroAction
    event_label: str


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


def _warm_extraction_policy(
    model: torch.nn.Module,
    *,
    device: torch.device,
) -> None:
    observation = ExtractionActorObservation(
        frame=np.zeros((84, 84), dtype=np.uint8),
        health=100,
        ammo=30,
        carried_value=0,
        free_slots=3,
        minimum_slot_value=0,
        banked_value=0,
        extraction_open=False,
        extraction_progress=0,
        remaining_time=75,
        previous_action=int(MacroAction.IDLE),
    )
    model(
        *extraction_observation_tensors(
            observation,
            episode_start=True,
            device=device,
        ),
        model.initial_state(1, device=device),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
        f"AMMO  {observation.ammo:.0f} / 30",
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
    policy_model: torch.nn.Module | None = None,
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
    if policy_model is None:
        model, _ = load_extraction_policy(
            checkpoint,
            style=style,
            scenario_hash=scenario_hash,
            device=device,
        )
    else:
        model = policy_model
    _warm_extraction_policy(model, device=device)
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
    frame_inputs: list[_ShowcaseFrameInput] = []
    event_counts: Counter[str] = Counter()
    hidden = model.initial_state(1, device=device)
    event_banner = "ENTER RAID  |  SEARCH FOR VALUE"
    banner_ttl = 0
    decisions = 0
    attack_decisions = 0
    route_cells: set[tuple[int, int]] = set()
    loot_region_cells: set[tuple[int, int]] = set()
    killed_opponent = False
    looted_cache_after_kill = False
    aggressive_chains = 0
    disengagement = DisengagementTracker()
    meaningful_extractions = 0
    backpack_upgrades = 0
    upgraded_backpack = False
    upgrade_to_extraction_conversions = 0
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
            if learner_action in {
                MacroAction.ATTACK,
                MacroAction.FORWARD_ATTACK,
                MacroAction.TURN_LEFT_ATTACK,
                MacroAction.TURN_RIGHT_ATTACK,
            }:
                attack_decisions += 1
            hidden = output.hidden
            state = env.privileged_state()
            x, y, opponent_x, opponent_y = (
                (state.host_x, state.host_y)
                if case.learner_side == "host"
                else (state.opponent_x, state.opponent_y)
            ) + (
                (state.opponent_x, state.opponent_y)
                if case.learner_side == "host"
                else (state.host_x, state.host_y)
            )
            route_cells.add((math.floor(x / 160), math.floor(y / 160)))
            opponent_distance = math.dist((x, y), (opponent_x, opponent_y))
            protocol_before = env.protocol_snapshot()
            disengagement.observe_encounter(
                health=observation.health,
                ammo=observation.ammo,
                opponent_distance=opponent_distance,
                learner_alive=(
                    protocol_before.public_state(case.learner_side).life_state == 1
                ),
                opponent_alive=(
                    protocol_before.public_state(opponent_side).life_state == 1
                ),
            )
            opponent_action = opponent.act(state)
            host_action, away_action = (
                (learner_action, opponent_action)
                if case.learner_side == "host"
                else (opponent_action, learner_action)
            )
            step = env.step(host_action, away_action)
            observations = type(observations)(step.host, step.opponent)
            state_after = env.privileged_state()
            after_x, after_y, after_opponent_x, after_opponent_y = (
                (
                    state_after.host_x,
                    state_after.host_y,
                    state_after.opponent_x,
                    state_after.opponent_y,
                )
                if case.learner_side == "host"
                else (
                    state_after.opponent_x,
                    state_after.opponent_y,
                    state_after.host_x,
                    state_after.host_y,
                )
            )
            protocol_after = env.protocol_snapshot()
            disengaged_now = disengagement.resolve_after_action(
                opponent_distance=math.dist(
                    (after_x, after_y),
                    (after_opponent_x, after_opponent_y),
                ),
                learner_alive=(
                    protocol_after.public_state(case.learner_side).life_state == 1
                ),
                opponent_alive=(
                    protocol_after.public_state(opponent_side).life_state == 1
                ),
            )
            for event in step.events:
                event_counts[f"{event.side}:{event.type.value}"] += event.count
            learner_events = tuple(
                event for event in step.events if event.side == case.learner_side
            )
            opponent_events = tuple(
                event for event in step.events if event.side == opponent_side
            )
            if any(
                event.type is ExtractionEventType.DEATH for event in opponent_events
            ):
                killed_opponent = True
            if any(
                event.type is ExtractionEventType.LOOT_PICKUP
                for event in learner_events
            ):
                loot_region_cells.add(
                    (math.floor(x / 160), math.floor(y / 160))
                )
            if any(
                event.type is ExtractionEventType.LOOT_DROP
                for event in learner_events
            ):
                backpack_upgrades += 1
                upgraded_backpack = True
            if (
                killed_opponent
                and any(
                    event.type is ExtractionEventType.CACHE_LOOTED
                    for event in learner_events
                )
            ):
                looted_cache_after_kill = True
            if any(
                event.type is ExtractionEventType.EXTRACTED
                for event in learner_events
            ):
                if observation.carried_value >= 25:
                    meaningful_extractions += 1
                if looted_cache_after_kill:
                    aggressive_chains += 1
                if upgraded_backpack:
                    upgrade_to_extraction_conversions += 1
            label = _event_label(step.events, learner_side=case.learner_side)
            if disengaged_now:
                label = "DISENGAGED  SAFE DISTANCE CREATED" + (
                    f" | {label}" if label else ""
                )
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
                frame_inputs.append(
                    _ShowcaseFrameInput(
                        observation=learner_observation,
                        privileged=env.privileged_state(),
                        protocol=env.protocol_snapshot(),
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
        env.close()
        frames = tuple(
            compose_extraction_showcase_frame(
                item.observation,
                privileged=item.privileged,
                protocol=item.protocol,
                learner_side=case.learner_side,
                style=style,
                action=item.action,
                event_label=item.event_label,
            )
            for item in frame_inputs
        )
        return RecordedExtractionShowcase(
            style=style,
            case=case,
            frames=frames,
            decisions=decisions,
            extracted_value=final.banked_value,
            extracted=life == 3,
            died=life == 2,
            events=dict(sorted(event_counts.items())),
            attack_decisions=attack_decisions,
            unique_route_cells=len(route_cells),
            aggressive_chains=aggressive_chains,
            successful_disengagements=disengagement.successes,
            meaningful_extractions=meaningful_extractions,
            meaningful_loot_regions=len(loot_region_cells),
            backpack_upgrades=backpack_upgrades,
            upgrade_to_extraction_conversions=upgrade_to_extraction_conversions,
            scenario_hash=reset_info.scenario_hash,
        )
    finally:
        env.close()
