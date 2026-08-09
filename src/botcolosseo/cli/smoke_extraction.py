from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_layouts import randomized_layout_variant
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a real synchronous randomized Extraction smoke"
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--decisions", type=int, default=10)
    parser.add_argument(
        "--host-action",
        type=int,
        default=int(MacroAction.IDLE),
    )
    parser.add_argument(
        "--opponent-action",
        type=int,
        default=int(MacroAction.IDLE),
    )
    args = parser.parse_args(argv)
    if args.decisions <= 0:
        raise ValueError("Extraction smoke decisions must be positive")
    root = repository_root()
    env = SynchronousExtractionEnv(
        config_path=root
        / "assets/scenarios/crystal_run_extraction_randomized/"
        "crystal_run_extraction_randomized.cfg",
        seed=args.seed,
        layout_variant=randomized_layout_variant(args.seed),
    )
    try:
        observations, info = env.reset()
        start = env.privileged_state()
        step = None
        for _ in range(args.decisions):
            step = env.step(
                MacroAction(args.host_action),
                MacroAction(args.opponent_action),
            )
            if step.terminated or step.truncated:
                break
        if step is None:
            raise RuntimeError("Extraction smoke produced no step")
        privileged = env.privileged_state()
        snapshot = env.protocol_snapshot()
        payload = {
            "ammo": {
                "host": observations.host.ammo,
                "opponent": observations.opponent.ammo,
            },
            "engine_tic": step.engine_tic,
            "inventory": {
                "host": {
                    "carried": observations.host.carried_value,
                    "free_slots": observations.host.free_slots,
                },
                "opponent": {
                    "carried": observations.opponent.carried_value,
                    "free_slots": observations.opponent.free_slots,
                },
            },
            "life_health": {
                "host": privileged.host_health,
                "opponent": privileged.opponent_health,
            },
            "loot_mask": privileged.world_loot_mask,
            "protocol": {
                "event_serial": snapshot.event_serial,
                "host_loot_pickups": snapshot.host_loot_pickups,
                "opponent_loot_pickups": snapshot.opponent_loot_pickups,
            },
            "peer_tic_lag": step.peer_tic_lag,
            "positions": {
                "start": [
                    start.host_x,
                    start.host_y,
                    start.opponent_x,
                    start.opponent_y,
                ],
                "end": [
                    privileged.host_x,
                    privileged.host_y,
                    privileged.opponent_x,
                    privileged.opponent_y,
                ],
            },
            "protocol_version": info.protocol_version,
            "round_state": privileged.round_state,
            "scenario_hash": info.scenario_hash,
            "terminated": step.terminated,
            "truncated": step.truncated,
        }
        if (
            payload["protocol_version"] != 3
            or payload["round_state"] != 1
            or payload["loot_mask"] != 127
            or payload["ammo"] != {"host": 30.0, "opponent": 30.0}
            or payload["peer_tic_lag"] > 2
        ):
            raise RuntimeError(f"Extraction smoke invariant failed: {payload}")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        env.close()
