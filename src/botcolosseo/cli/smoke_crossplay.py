from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from botcolosseo.agents.league_opponents import OpponentSpec, sha256_file
from botcolosseo.cli.evaluate_crossplay import (
    CrossplayControllerFactory,
    run_case_with_retries,
)
from botcolosseo.evaluation.crossplay import run_crossplay_episode
from botcolosseo.scenarios.league_splits import load_league_cases
from botcolosseo.scenarios.regions import RegionGraph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke a learned M3 policy on both duel sides")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--max-decisions", type=int, default=525)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 < args.pairs <= 5 or args.max_decisions <= 0:
        raise ValueError("Cross-play smoke limits are invalid")
    root = Path(__file__).resolve().parents[3]
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    scenario_hash = json.loads(
        (root / "assets/scenarios/crystal_run/manifest.json").read_text(
            encoding="utf-8"
        )
    )["wad_sha256"]
    spec = OpponentSpec(
        opponent_id="smoke-policy",
        kind="checkpoint",
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        scenario_hash=scenario_hash,
        selection_evidence="development:crossplay-smoke",
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    graph = RegionGraph.from_yaml(
        root / "assets/scenarios/crystal_run/src/regions.yaml"
    )
    factory = CrossplayControllerFactory(graph=graph, device=device)
    cases = load_league_cases(
        root / "configs/m3/validation.json",
        expected_split="validation",
        expected_pairs=50,
    )[: args.pairs * 2]
    rows = []
    for case in cases:
        left_side = case.learner_side
        right_side = "opponent" if left_side == "host" else "host"
        rows.append(
            run_case_with_retries(
                lambda case=case, left_side=left_side, right_side=right_side: run_crossplay_episode(
                    case,
                    left_spec=spec,
                    right_spec=spec,
                    left_controller=factory.create(spec, side=left_side),
                    right_controller=factory.create(spec, side=right_side),
                    graph=graph,
                    config_path=root
                    / "assets/scenarios/crystal_run/crystal_run.cfg",
                    max_decisions=args.max_decisions,
                ),
                max_attempts=2,
            )
        )
    protocol_inconsistencies = sum(
        row.protocol_inconsistent
        or row.action_tic_inconsistent
        or row.score_event_inconsistent
        or row.peer_tic_lag_max != 0
        for row in rows
    )
    result = {
        "checkpoint_sha256": spec.checkpoint_sha256,
        "gate": "PASS" if protocol_inconsistencies == 0 else "FAIL",
        "pairs": args.pairs,
        "protocol_inconsistencies": protocol_inconsistencies,
        "rows": [row.to_dict() for row in rows],
        "scenario_hash": scenario_hash,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate"] == "PASS" else 1
