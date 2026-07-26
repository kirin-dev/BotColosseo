from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from botcolosseo.agents.extraction_teachers import (
    AggressiveExtractionTeacher,
    ExtractionWaypointTeacher,
)
from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_protocol import ExtractionEventType
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _event_counts(counter: Counter[ExtractionEventType]) -> dict[str, int]:
    return {
        event_type.value: counter[event_type]
        for event_type in ExtractionEventType
        if counter[event_type]
    }


def _route_and_replacement(seed: int) -> dict[str, object]:
    root = repository_root()
    env = SynchronousExtractionEnv(
        config_path=root
        / "assets/scenarios/crystal_run_extraction/crystal_run_extraction.cfg",
        seed=seed,
    )
    host = ExtractionWaypointTeacher(
        side="host",
        waypoints=(
            (-520.0, 288.0),
            (-520.0, -288.0),
            (-224.0, 0.0),
            (224.0, 0.0),
            (0.0, 0.0),
            (0.0, 400.0),
        ),
        arrival_tolerance=28.0,
    )
    opponent = ExtractionWaypointTeacher(
        side="opponent",
        waypoints=(
            (520.0, -288.0),
            (320.0, -288.0),
            (96.0, -320.0),
            (0.0, -400.0),
        ),
        arrival_tolerance=28.0,
    )
    counts: Counter[ExtractionEventType] = Counter()
    try:
        env.reset()
        for index in range(700):
            decisions = index + 1
            state = env.privileged_state()
            step = env.step(host.act(state), opponent.act(state))
            for event in step.events:
                counts[event.type] += event.count
            if step.terminated or step.truncated:
                break
        else:
            raise RuntimeError("Replacement route exceeded the decision budget")
        state = env.privileged_state()
        if (
            not step.terminated
            or step.truncated
            or step.winner != 1
            or state.host_banked != 100
            or counts[ExtractionEventType.LOOT_DROP] < 2
            or counts[ExtractionEventType.EXTRACTED] != 2
        ):
            raise RuntimeError(
                "Replacement route failed: "
                f"decisions={decisions}, engine_tic={step.engine_tic}, "
                f"terminated={step.terminated}, truncated={step.truncated}, "
                f"winner={step.winner}, positions="
                f"{(state.host_x, state.host_y, state.opponent_x, state.opponent_y)}, "
                f"slots={(state.host_slots, state.opponent_slots)}, "
                f"mask={state.world_loot_mask}, banked={state.host_banked}, "
                f"events={_event_counts(counts)}"
            )
        return {
            "decisions": decisions,
            "engine_tic": step.engine_tic,
            "event_counts": _event_counts(counts),
            "host_banked": state.host_banked,
            "opponent_banked": state.opponent_banked,
            "winner": step.winner,
        }
    finally:
        env.close()


def _combat_cache_extract(seed: int) -> dict[str, object]:
    root = repository_root()
    env = SynchronousExtractionEnv(
        config_path=root
        / "assets/scenarios/crystal_run_extraction/crystal_run_extraction.cfg",
        seed=seed,
    )
    opponent_route = ExtractionWaypointTeacher(
        side="opponent",
        waypoints=((520.0, 288.0), (224.0, 0.0)),
    )
    aggressive = AggressiveExtractionTeacher(side="host")
    counts: Counter[ExtractionEventType] = Counter()
    health_trace: list[int] = [100]
    position_trace: list[tuple[int, float, float, float, int]] = []
    preparation_trace: list[tuple[int, float, float, float, float]] = []
    decisions = 0
    try:
        env.reset()
        while sum(env.privileged_state().opponent_slots) < 35:
            decisions += 1
            state = env.privileged_state()
            step = env.step(MacroAction.IDLE, opponent_route.act(state))
            if decisions == 1 or decisions % 10 == 0:
                current = env.privileged_state()
                preparation_trace.append(
                    (
                        decisions,
                        round(current.host_x, 1),
                        round(current.host_y, 1),
                        round(current.host_health, 1),
                        round(current.opponent_x, 1),
                    )
                )
            for event in step.events:
                counts[event.type] += event.count
            if decisions >= 250:
                raise RuntimeError("Opponent did not collect combat cache loot")

        while env.privileged_state().opponent_health > 0:
            decisions += 1
            state = env.privileged_state()
            action = aggressive.act(state)
            if not position_trace or decisions % 25 == 0:
                position_trace.append(
                    (
                        decisions,
                        round(state.host_x, 1),
                        round(state.host_y, 1),
                        round(state.host_angle, 1),
                        int(action),
                    )
                )
            step = env.step(action, MacroAction.IDLE)
            for event in step.events:
                counts[event.type] += event.count
            health = int(step.opponent.health)
            if health != health_trace[-1]:
                health_trace.append(health)
            if decisions >= 450:
                current = env.privileged_state()
                protocol = env.protocol_snapshot()
                raise RuntimeError(
                    "Aggressive teacher did not finish combat: "
                    f"engine={current.engine_tic}, "
                    f"host_position={(current.host_x, current.host_y)}, "
                    f"opponent_position={(current.opponent_x, current.opponent_y)}, "
                    f"angles={(current.host_angle, current.opponent_angle)}, "
                    f"life={(protocol.host_life_state, protocol.opponent_life_state)}, "
                    f"health={(current.host_health, current.opponent_health)}, "
                    f"ammo={(step.host.ammo, step.opponent.ammo)}, "
                    f"trace={health_trace}, positions_trace={position_trace}, "
                    f"preparation_trace={preparation_trace}, "
                    f"events={_event_counts(counts)}"
                )

        cache = env.privileged_state()
        cache_route = ExtractionWaypointTeacher(
            side="host",
            waypoints=((cache.cache_x, cache.cache_y),),
        )
        while counts[ExtractionEventType.CACHE_LOOTED] < 1:
            decisions += 1
            state = env.privileged_state()
            step = env.step(cache_route.act(state), MacroAction.IDLE)
            for event in step.events:
                counts[event.type] += event.count
            if decisions >= 550:
                current = env.privileged_state()
                raise RuntimeError(
                    "Aggressive teacher did not upgrade from the corpse cache: "
                    f"host_position={(current.host_x, current.host_y)}, "
                    f"cache_position={(current.cache_x, current.cache_y)}, "
                    f"slots={(current.host_slots, current.cache_slots)}, "
                    f"cache_owner={current.cache_owner}, events={_event_counts(counts)}"
                )

        extraction = ExtractionWaypointTeacher(
            side="host",
            waypoints=((0.0, 400.0),),
            arrival_tolerance=28.0,
        )
        while True:
            decisions += 1
            state = env.privileged_state()
            step = env.step(extraction.act(state), MacroAction.IDLE)
            for event in step.events:
                counts[event.type] += event.count
            if step.terminated or step.truncated:
                break
            if decisions >= 700:
                raise RuntimeError("Aggressive teacher did not extract")

        state = env.privileged_state()
        if (
            not step.terminated
            or step.truncated
            or step.winner != 1
            or health_trace != [100, 80, 60, 40, 20, 0]
            or counts[ExtractionEventType.VALID_HIT] != 5
            or counts[ExtractionEventType.CACHE_CREATED] != 1
            or counts[ExtractionEventType.CACHE_LOOTED] < 1
            or counts[ExtractionEventType.EXTRACTED] != 1
            or state.host_banked <= 0
        ):
            raise RuntimeError(
                "Combat-cache-extract chain failed: "
                f"decisions={decisions}, engine_tic={step.engine_tic}, "
                f"position={(state.host_x, state.host_y)}, "
                f"extraction={(step.host.extraction_open, step.host.extraction_progress)}, "
                f"winner={step.winner}, banked={state.host_banked}, "
                f"slots={state.host_slots}, "
                f"health={health_trace}, events={_event_counts(counts)}"
            )
        return {
            "decisions": decisions,
            "engine_tic": step.engine_tic,
            "event_counts": _event_counts(counts),
            "health_trace": health_trace,
            "host_banked": state.host_banked,
            "winner": step.winner,
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the real Crystal Run Extraction v2 X0 mechanics gate"
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = repository_root()
    config_path = (
        root / "assets/scenarios/crystal_run_extraction/crystal_run_extraction.cfg"
    )
    scenario_manifest = json.loads(
        (config_path.parent / "manifest.json").read_text(encoding="utf-8")
    )
    result = {
        "config_sha256": sha256_file(config_path),
        "mechanics": {
            "backpack_slots": 3,
            "extraction_zone_count": 2,
            "fixed_damage": 20,
            "starting_ammo": 30,
            "terminal_respawns": 0,
        },
        "mechanics_gate_passed": True,
        "protocol_version": 3,
        "schema_version": 1,
        "scenario_hash": scenario_manifest["wad_sha256"],
        "stage": "extraction-v2-x0-mechanics",
        "route_and_replacement": _route_and_replacement(args.seed),
        "combat_cache_extract": _combat_cache_extract(args.seed + 1),
        "test_cases_accessed": False,
    }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite X0 report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
