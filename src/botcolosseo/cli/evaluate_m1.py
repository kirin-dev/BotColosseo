from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from botcolosseo.envs.task_runner import run_teacher_episode
from botcolosseo.evaluation.m1 import (
    M1_TEACHERS,
    evaluate_cases,
    load_cases,
    write_evidence,
)
from botcolosseo.scenarios.splits import EpisodeCase


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen Milestone 1 gate")
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args(argv)
    if args.max_cases is not None and not args.development:
        parser.error("--max-cases requires --development")
    if args.max_cases is not None and args.max_cases <= 0:
        parser.error("--max-cases must be positive")

    root = Path(__file__).resolve().parents[3]
    config_paths = {
        name: root / "configs/m1" / f"{name}.json"
        for name in ("train", "validation", "test")
    }
    all_cases = {name: load_cases(path, name) for name, path in config_paths.items()}
    expected_counts = Counter({task: 100 for task in M1_TEACHERS})
    for name, split_cases in all_cases.items():
        counts = Counter(case.task for case in split_cases)
        if counts != expected_counts:
            raise ValueError(f"M1 {name} manifest must contain 100 cases per task")
    seed_sets = [{case.seed for case in all_cases[name]} for name in all_cases]
    if any(seed_sets[left] & seed_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("M1 split manifests contain overlapping seeds")
    cases = all_cases[args.split]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    official = args.split == "test" and not args.development and len(cases) == 500

    def runner(case: EpisodeCase, teacher: str):
        return run_teacher_episode(task=case.task, teacher_name=teacher, seed=case.seed)

    def progress(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            print(f"M1 progress: {done}/{total}", flush=True)

    summary, rows = evaluate_cases(cases, runner=runner, official=official, progress=progress)
    scenario_manifest = root / "assets/scenarios/crystal_run/manifest.json"
    manifest = {
        "official": official,
        "split": args.split,
        "episodes": len(rows),
        "git_commit": _git_commit(root),
        "split_sha256": _sha256(config_paths[args.split]),
        "scenario_manifest_sha256": _sha256(scenario_manifest),
    }
    write_evidence(args.output, summary, rows, manifest)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0 if summary.passed or args.development else 1
