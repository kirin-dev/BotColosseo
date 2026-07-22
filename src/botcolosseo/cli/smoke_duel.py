from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import numpy as np

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.synchronous_duel import SynchronousDuelEnv
from botcolosseo.scenarios.regions import RegionGraph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke the synchronous Crystal Run duel")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--decisions", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.decisions <= 0:
        raise ValueError("--decisions must be positive")
    root = Path(__file__).resolve().parents[3]
    graph = RegionGraph.from_yaml(root / "assets/scenarios/crystal_run/src/regions.yaml")
    before = {child.pid for child in mp.active_children()}
    env = SynchronousDuelEnv(
        config_path=root / "assets/scenarios/crystal_run/crystal_run.cfg",
        region_graph=graph,
        seed=args.seed,
        max_decisions=args.decisions,
    )
    rng = np.random.default_rng(args.seed)
    events: Counter[str] = Counter()
    completed = 0
    info = None
    last_tic = 0
    terminated = False
    truncated = False
    try:
        _, info = env.reset()
        last_tic = info.engine_tic
        while completed < args.decisions and not terminated and not truncated:
            host_action = MacroAction(int(rng.integers(0, len(MacroAction))))
            opponent_action = MacroAction(int(rng.integers(0, len(MacroAction))))
            step = env.step(host_action, opponent_action)
            if step.engine_tic != last_tic + 4:
                raise RuntimeError("Duel smoke observed a non-four-tic decision")
            last_tic = step.engine_tic
            completed += 1
            events.update(f"{event.side}:{event.type.value}" for event in step.events)
            terminated = step.terminated
            truncated = step.truncated
    finally:
        env.close()
    cleaned = {child.pid for child in mp.active_children()} <= before
    summary = {
        "cleaned_workers": cleaned,
        "completed_decisions": completed,
        "engine_tic": last_tic,
        "event_counts": dict(sorted(events.items())),
        "port": None if info is None else info.port,
        "protocol_version": None if info is None else info.protocol_version,
        "scenario_hash": None if info is None else info.scenario_hash,
        "seed": args.seed,
        "terminated": terminated,
        "truncated": truncated,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if cleaned and completed > 0 else 1
