from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.agents.duel_teachers import DUEL_TEACHERS, create_duel_teacher
from botcolosseo.envs.synchronous_duel import SynchronousDuelEnv
from botcolosseo.evaluation.sync_audit import run_sync_audit, write_audit_summary
from botcolosseo.scenarios.regions import RegionGraph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the M2 1v1 synchronization audit"
    )
    parser.add_argument("--decisions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--host-teacher", choices=sorted(DUEL_TEACHERS), default="aggressive_script"
    )
    parser.add_argument(
        "--opponent-teacher",
        choices=sorted(DUEL_TEACHERS),
        default="objective_first",
    )
    parser.add_argument("--video-frame-cap", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("reports/m2/sync-audit.json"))
    parser.add_argument(
        "--video", type=Path, default=Path("docs/assets/m2-sync-duel.mp4")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    output = args.output if args.output.is_absolute() else root / args.output
    video = args.video if args.video.is_absolute() else root / args.video
    graph = RegionGraph.from_yaml(
        root / "assets/scenarios/crystal_run/src/regions.yaml"
    )
    env = SynchronousDuelEnv(
        config_path=root / "assets/scenarios/crystal_run/crystal_run.cfg",
        region_graph=graph,
        seed=args.seed,
    )
    summary = run_sync_audit(
        env,
        create_duel_teacher(args.host_teacher, graph, side="host"),
        create_duel_teacher(args.opponent_teacher, graph, side="opponent"),
        decisions=args.decisions,
        seed=args.seed,
        video_path=video,
        video_frame_cap=args.video_frame_cap,
    )
    write_audit_summary(summary, output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1
