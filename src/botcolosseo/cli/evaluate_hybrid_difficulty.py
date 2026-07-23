from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import torch

from botcolosseo.agents.difficulty import load_difficulty_profiles
from botcolosseo.agents.hybrid_config import load_hybrid_policy_config
from botcolosseo.agents.hybrid_difficulty import TracedHybridDifficultyPolicy
from botcolosseo.agents.hybrid_policy import build_hybrid_style_policy
from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.cli.evaluate_style import select_style_cases
from botcolosseo.cli.train_ppo import _atomic_json
from botcolosseo.evaluation.defensive import run_defensive_episode
from botcolosseo.evaluation.explorer import run_explorer_episode
from botcolosseo.evaluation.hybrid_difficulty import (
    HYBRID_DIFFICULTIES,
    HybridDifficultyExecutionRow,
    HybridDifficultyGovernorRow,
    evaluate_hybrid_difficulty_extension,
)
from botcolosseo.evaluation.hybrid_difficulty_config import (
    load_hybrid_difficulty_product_config,
)
from botcolosseo.evaluation.m2 import load_duel_cases
from botcolosseo.evaluation.native_style_difficulty import (
    NativeEpisode,
    NativeStyle,
    NativeStyleDifficultyRecord,
)
from botcolosseo.scenarios.duel_splits import DuelCase
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.training.bc import append_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a hybrid style on frozen Easy/Normal difficulty"
    )
    parser.add_argument("--style", choices=("defensive", "explorer"), required=True)
    parser.add_argument(
        "--product-config",
        type=Path,
        default=Path("configs/m5/hybrid/difficulty-product.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs-per-opponent", type=int, default=10)
    parser.add_argument("--max-decisions", type=int, default=525)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.pairs_per_opponent, args.max_decisions, args.max_attempts) <= 0:
        raise ValueError("Hybrid difficulty settings must be positive")
    style: NativeStyle = args.style
    root = Path(__file__).resolve().parents[3]
    product = load_hybrid_difficulty_product_config(
        args.product_config,
        root=root,
    )
    source = product.defensive if style == "defensive" else product.explorer
    hybrid = load_hybrid_policy_config(source.governor_config, root=root)
    if hybrid.style != style or hybrid.scenario_hash != product.scenario_hash:
        raise ValueError("Hybrid difficulty style source identity is invalid")
    profiles = load_difficulty_profiles(product.difficulty_config)
    selected_profiles = {
        difficulty: profiles[difficulty] for difficulty in HYBRID_DIFFICULTIES
    }
    cases = load_duel_cases(
        product.cases,
        expected_split="validation",
        pairs_per_opponent=50,
    )
    selected = select_style_cases(
        cases,
        pairs_per_opponent=args.pairs_per_opponent,
    )
    output_dir = _resolve(root, args.output_dir)
    identity = {
        "schema_version": 1,
        "stage": f"m5-hybrid-{style}-difficulty-extension",
        "style": style,
        "split": "validation",
        "product_config_sha256": product.config_sha256,
        "governor_config_sha256": hybrid.config_sha256,
        "base_checkpoint_sha256": hybrid.base_checkpoint_sha256,
        "difficulty_config_sha256": product.difficulty_config_sha256,
        "cases_sha256": product.cases_sha256,
        "selected_case_ids": [
            [case.opponent, case.pair_index, case.learner_side] for case in selected
        ],
        "scenario_hash": product.scenario_hash,
        "code_revision": _git_revision(root),
        "difficulties": list(HYBRID_DIFFICULTIES),
        "profiles": {
            name: {
                "reaction_delay": profile.reaction_delay,
                "policy_update_interval": profile.policy_update_interval,
            }
            for name, profile in selected_profiles.items()
        },
        "pairs_per_opponent": args.pairs_per_opponent,
        "max_decisions": args.max_decisions,
        "max_attempts": args.max_attempts,
        "expected_episodes": len(HYBRID_DIFFICULTIES) * len(selected),
        "test_cases_accessed": False,
    }
    if args.preflight:
        print(json.dumps({**identity, "preflight_passed": True}, indent=2, sort_keys=True))
        return 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.json"
    if run_path.exists():
        if _json(run_path) != identity:
            raise ValueError("Hybrid difficulty resume identity does not match")
    else:
        _atomic_json(identity, run_path)
    episodes_path = output_dir / "episodes.jsonl"
    telemetry_path = output_dir / "telemetry.jsonl"
    trace_path = output_dir / "execution-trace.jsonl"
    records = _load_records(episodes_path, style=style)
    governor_rows = _load_governor(telemetry_path)
    execution_rows = _load_execution(trace_path)
    _validate_resume_prefix(
        records,
        governor_rows,
        execution_rows,
        selected=selected,
        profiles=selected_profiles,
    )
    policies = {
        difficulty: TracedHybridDifficultyPolicy(
            style,
            build_hybrid_style_policy(hybrid, device=device),
            selected_profiles[difficulty],
        )
        for difficulty in HYBRID_DIFFICULTIES
    }
    for difficulty, policy in policies.items():
        for _ in range(sum(row.difficulty == difficulty for row in records)):
            policy.reset(seed=0)
    graph = RegionGraph.from_yaml(
        root / "assets/scenarios/crystal_run/src/regions.yaml"
    )
    arena = root / "assets/scenarios/crystal_run/crystal_run.cfg"
    completed = {
        (
            row.difficulty,
            row.episode.opponent,
            row.episode.pair_index,
            row.episode.learner_side,
        )
        for row in records
    }
    total = int(identity["expected_episodes"])
    runner = _runner(style)
    for difficulty in HYBRID_DIFFICULTIES:
        policy = policies[difficulty]
        for case in selected:
            key = (difficulty, case.opponent, case.pair_index, case.learner_side)
            if key in completed:
                continue
            episode = _run_with_retries(
                lambda selected_case=case, selected_policy=policy: runner(
                    selected_case,
                    policy=selected_policy,
                    graph=graph,
                    config_path=arena,
                    max_decisions=args.max_decisions,
                ),
                max_attempts=args.max_attempts,
            )
            record = NativeStyleDifficultyRecord(difficulty, episode)
            telemetry, trace = policy.drain_evidence()
            if len(trace) != episode.decisions:
                raise RuntimeError("Hybrid difficulty episode trace is incomplete")
            episode_governor = [
                HybridDifficultyGovernorRow.from_telemetry(
                    style=style,
                    difficulty=difficulty,
                    opponent=case.opponent,
                    pair_index=case.pair_index,
                    learner_side=case.learner_side,
                    telemetry=row,
                )
                for row in telemetry
            ]
            episode_execution = [
                HybridDifficultyExecutionRow.from_trace(
                    style=style,
                    difficulty=difficulty,
                    opponent=case.opponent,
                    pair_index=case.pair_index,
                    learner_side=case.learner_side,
                    trace=row,
                )
                for row in trace
            ]
            append_jsonl(episodes_path, record.to_dict())
            for row in episode_governor:
                append_jsonl(telemetry_path, row.to_dict())
            for row in episode_execution:
                append_jsonl(trace_path, row.to_dict())
            records.append(record)
            governor_rows.extend(episode_governor)
            execution_rows.extend(episode_execution)
            completed.add(key)
            print(
                f"M5 hybrid {style} difficulty progress: {len(records)}/{total}",
                flush=True,
            )
    maximum = hybrid.governor.max_consecutive_interventions
    summary = evaluate_hybrid_difficulty_extension(
        records,
        governor_rows,
        execution_rows,
        style=style,
        profiles=selected_profiles,
        max_consecutive_interventions=maximum,
        expected_pairs_per_opponent=args.pairs_per_opponent,
        expected_scenario_hash=product.scenario_hash,
    )
    payload = {
        **summary,
        "split": "validation",
        "product_config_sha256": product.config_sha256,
        "governor_config_sha256": hybrid.config_sha256,
        "base_checkpoint_sha256": hybrid.base_checkpoint_sha256,
        "difficulty_config_sha256": product.difficulty_config_sha256,
        "cases_sha256": product.cases_sha256,
        "scenario_hash": product.scenario_hash,
        "test_cases_accessed": False,
    }
    summary_path = output_dir / "summary.json"
    _atomic_json(payload, summary_path)
    _atomic_json(
        {
            **identity,
            "episodes": len(records),
            "governor_decisions": len(governor_rows),
            "executed_decisions": len(execution_rows),
            "episodes_sha256": sha256_file(episodes_path),
            "telemetry_sha256": sha256_file(telemetry_path),
            "execution_trace_sha256": sha256_file(trace_path),
            "summary_sha256": sha256_file(summary_path),
            "passed": summary["passed"],
        },
        output_dir / "manifest.json",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


def _validate_resume_prefix(
    records: list[NativeStyleDifficultyRecord],
    governor_rows: list[HybridDifficultyGovernorRow],
    execution_rows: list[HybridDifficultyExecutionRow],
    *,
    selected: tuple[DuelCase, ...],
    profiles,
) -> None:
    expected = [
        (difficulty, case.opponent, case.pair_index, case.learner_side)
        for difficulty in HYBRID_DIFFICULTIES
        for case in selected
    ]
    actual = [
        (
            row.difficulty,
            row.episode.opponent,
            row.episode.pair_index,
            row.episode.learner_side,
        )
        for row in records
    ]
    if actual != expected[: len(actual)]:
        raise ValueError("Hybrid difficulty ledger is not an ordered resume prefix")
    completed = set(actual)
    if (
        {row.episode_key for row in governor_rows} != completed
        or {row.episode_key for row in execution_rows} != completed
    ):
        if completed or governor_rows or execution_rows:
            raise ValueError("Hybrid difficulty resume evidence is incomplete")
    decisions = {
        key: record.episode.decisions
        for key, record in zip(actual, records, strict=True)
    }
    governor_counts = Counter(row.episode_key for row in governor_rows)
    execution_counts = Counter(row.episode_key for row in execution_rows)
    if any(
        execution_counts[key] != count
        or governor_counts[key]
        != (count + profiles[key[0]].policy_update_interval - 1)
        // profiles[key[0]].policy_update_interval
        for key, count in decisions.items()
    ):
        raise ValueError("Hybrid difficulty resume evidence counts drifted")


def _load_records(
    path: Path,
    *,
    style: NativeStyle,
) -> list[NativeStyleDifficultyRecord]:
    if not path.exists():
        return []
    return [
        NativeStyleDifficultyRecord.from_dict(json.loads(line), style=style)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_governor(path: Path) -> list[HybridDifficultyGovernorRow]:
    if not path.exists():
        return []
    return [
        HybridDifficultyGovernorRow.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_execution(path: Path) -> list[HybridDifficultyExecutionRow]:
    if not path.exists():
        return []
    return [
        HybridDifficultyExecutionRow.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _runner(style: NativeStyle):
    return run_defensive_episode if style == "defensive" else run_explorer_episode


def _run_with_retries(
    runner: Callable[[], NativeEpisode],
    *,
    max_attempts: int,
) -> NativeEpisode:
    for attempt in range(1, max_attempts + 1):
        try:
            return replace(runner(), environment_attempts=attempt)
        except RuntimeError as error:
            retriable = str(error) == (
                "Duel respawn did not complete within the warm-up limit"
            )
            if not retriable or attempt == max_attempts:
                raise
    raise AssertionError("unreachable")


def _resolve(root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if len(revision) != 40:
        raise ValueError("Could not bind hybrid difficulty to a Git revision")
    return revision


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload
