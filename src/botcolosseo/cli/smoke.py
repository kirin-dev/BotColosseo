from __future__ import annotations

import argparse
from pathlib import Path

import vizdoom as vzd

from botcolosseo.envs.smoke import run_smoke
from botcolosseo.envs.vizdoom_game import GameSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Bot Colosseo ViZDoom smoke gate")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(vzd.scenarios_path) / "basic.cfg",
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-decisions", type=int, default=100)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--require-video", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_smoke(
        GameSettings(config_path=args.config, seed=args.seed),
        episodes=args.episodes,
        max_decisions=args.max_decisions,
        frame_skip=args.frame_skip,
        video_path=args.record,
        require_video=args.require_video,
    )
    print(summary.to_json())
    if not summary.all_terminated:
        return 1
    if args.require_video and summary.video_error is not None:
        return 1
    return 0
