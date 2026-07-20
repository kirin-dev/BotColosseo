from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from botcolosseo.demo.m2_showcase import (
    ShowcaseEpisode,
    compose_policy_comparison,
    record_showcase_episode,
)
from botcolosseo.envs.video import write_mp4
from botcolosseo.evaluation.m2 import load_actor_policy, load_duel_cases, sha256_file
from botcolosseo.scenarios.duel_splits import DuelCase
from botcolosseo.scenarios.regions import RegionGraph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render selected M2 policies")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--max-decisions", type=int, default=525)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/assets/m2-policy-comparison.mp4")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/m2/policy-comparison.json")
    )
    return parser


def _run_with_retry(runner: Callable[[], ShowcaseEpisode]) -> ShowcaseEpisode:
    try:
        return runner()
    except RuntimeError as error:
        if str(error) != "Duel respawn did not complete within the warm-up limit":
            raise
        return runner()


def load_showcase_case(path: Path, case_index: int) -> DuelCase:
    cases = load_duel_cases(
        path,
        expected_split="validation",
        pairs_per_opponent=50,
    )
    try:
        return cases[case_index]
    except IndexError as error:
        raise ValueError("--case-index is outside the validation manifest") from error


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.case_index < 0 or args.max_decisions <= 0 or args.fps <= 0:
        raise ValueError("Showcase numeric arguments must be positive")
    root = Path(__file__).resolve().parents[3]
    graph = RegionGraph.from_yaml(
        root / "assets/scenarios/crystal_run/src/regions.yaml"
    )
    case = load_showcase_case(root / "configs/m2/validation.json", args.case_index)
    if case.split != "validation":
        raise ValueError("Showcase case must belong to the validation split")
    scenario_hash = json.loads(
        (root / "assets/scenarios/crystal_run/manifest.json").read_text()
    )["wad_sha256"]
    device = torch.device(args.device)
    checkpoints = {
        "PPO": root / "runs/m2/ppo-full/selected.pt",
        "BC": root / "runs/m2/bc-full/best.pt",
    }
    streams: dict[str, list] = {}
    episode_summaries: dict[str, dict[str, object]] = {}
    for label, checkpoint in checkpoints.items():
        policy = load_actor_policy(
            label.lower(),
            checkpoint,
            device=device,
            expected_scenario_hash=scenario_hash,
        )
        episode = _run_with_retry(
            lambda policy=policy: record_showcase_episode(
                case,
                policy=policy,
                graph=graph,
                config_path=root / "assets/scenarios/crystal_run/crystal_run.cfg",
                max_decisions=args.max_decisions,
            )
        )
        streams[label] = list(episode.frames)
        episode_summaries[label.lower()] = {
            "decisions": episode.decisions,
            "learner_score": episode.learner_score,
            "objective_completed": episode.objective_completed,
            "opponent_score": episode.opponent_score,
            "terminated": episode.terminated,
            "truncated": episode.truncated,
        }
    subtitle = f"VALIDATION ONLY | seed={case.seed} | vs {case.opponent}"
    frames = compose_policy_comparison(streams, subtitle=subtitle)
    output = args.output if args.output.is_absolute() else root / args.output
    write_mp4(frames, output, fps=args.fps)
    report = {
        "case": case.to_dict(),
        "checkpoint_sha256": {
            label.lower(): sha256_file(path) for label, path in checkpoints.items()
        },
        "episodes": episode_summaries,
        "fps": args.fps,
        "frame_count": len(frames),
        "official_test_result": False,
        "scenario_hash": scenario_hash,
        "schema_version": 1,
        "split": "validation",
        "test_cases_accessed": False,
        "video": str(output.relative_to(root)),
        "video_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    report_path = args.report if args.report.is_absolute() else root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
