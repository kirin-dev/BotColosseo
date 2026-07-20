from __future__ import annotations

import argparse
from pathlib import Path

from botcolosseo.agents.teachers import TEACHER_REGISTRY
from botcolosseo.envs.task_runner import run_teacher_episode
from botcolosseo.scenarios.splits import TaskKind


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Crystal Run Teacher smoke episode")
    parser.add_argument("--task", choices=[task.value for task in TaskKind], required=True)
    parser.add_argument("--teacher", choices=sorted(TEACHER_REGISTRY), required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--require-video", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_teacher_episode(
        task=TaskKind(args.task),
        teacher_name=args.teacher,
        seed=args.seed,
        video_path=args.record,
        require_video=args.require_video,
    )
    print(summary.to_json())
    if not summary.success:
        return 1
    if args.require_video and summary.video_error is not None:
        return 1
    return 0
