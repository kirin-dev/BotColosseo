from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from botcolosseo.agents.duel_teachers import DUEL_TEACHERS
from botcolosseo.evaluation.m2 import (
    FROZEN_M2_THRESHOLDS,
    M2_POLICIES,
    M2EpisodeRecord,
    TeacherEvaluationPolicy,
    evaluate_m2_records,
    load_actor_policy,
    load_duel_cases,
    run_m2_episode,
    sha256_file,
    write_m2_evidence,
)
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS, DuelCase
from botcolosseo.scenarios.regions import RegionGraph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen Milestone 2 evaluator")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/m2/evaluation.yaml")
    )
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--policies", nargs="+")
    parser.add_argument("--opponents", nargs="+")
    parser.add_argument("--ppo-checkpoint", type=Path)
    parser.add_argument("--bc-checkpoint", type=Path)
    return parser


def _git_provenance(root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    return commit, bool(status.strip())


def _resolve(root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _selected_cases(
    cases: Sequence[DuelCase],
    *,
    opponents: Sequence[str],
    max_pairs: int | None,
) -> tuple[DuelCase, ...]:
    selected: list[DuelCase] = []
    for opponent in opponents:
        opponent_cases = [case for case in cases if case.opponent == opponent]
        if max_pairs is not None:
            opponent_cases = opponent_cases[: max_pairs * 2]
        selected.extend(opponent_cases)
    return tuple(selected)


def ensure_evidence_targets_absent(output_dir: Path) -> None:
    targets = (
        "episodes.csv",
        "summary.json",
        "manifest.json",
        ".episodes.csv.tmp",
        ".summary.json.tmp",
        ".manifest.json.tmp",
    )
    conflicts = [name for name in targets if (output_dir / name).exists()]
    if conflicts:
        raise FileExistsError(
            f"Evaluation evidence already exists: {', '.join(conflicts)}"
        )


def run_case_with_retries(
    runner: Callable[[], M2EpisodeRecord], *, max_attempts: int
) -> M2EpisodeRecord:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    config_path = _resolve(root, args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported M2 evaluation config schema")
    if config.get("thresholds") != FROZEN_M2_THRESHOLDS:
        raise ValueError("M2 evaluation thresholds differ from the frozen implementation")
    if config.get("action_selection") != "greedy":
        raise ValueError("Official M2 action selection must be greedy")
    if args.split == "test" and args.development:
        parser.error("The frozen test split cannot be used in development mode")
    if args.max_pairs is not None and (
        not args.development or args.max_pairs <= 0
    ):
        parser.error("--max-pairs must be positive and requires --development")
    if args.max_decisions is not None and (
        not args.development or args.max_decisions <= 0
    ):
        parser.error("--max-decisions must be positive and requires --development")

    official_policies = tuple(config["official_policies"])
    policies = tuple(args.policies or official_policies)
    opponents = tuple(args.opponents or config["opponents"])
    valid_policy_names = set(DUEL_TEACHERS) | set(M2_POLICIES)
    if not policies or any(policy not in valid_policy_names for policy in policies):
        parser.error("Unknown or empty evaluation policy selection")
    if not opponents or any(opponent not in DUEL_OPPONENTS for opponent in opponents):
        parser.error("Unknown or empty opponent selection")
    if len(policies) != len(set(policies)) or len(opponents) != len(set(opponents)):
        parser.error("Policy and opponent selections must be unique")

    split_spec = config["splits"][args.split]
    split_path = _resolve(root, split_spec["path"])
    if sha256_file(split_path) != split_spec["sha256"]:
        raise ValueError(f"Frozen {args.split} manifest hash does not match")
    pairs_per_opponent = int(config["pairs_per_opponent"])
    all_cases = load_duel_cases(
        split_path,
        expected_split=args.split,
        pairs_per_opponent=pairs_per_opponent,
    )
    cases = _selected_cases(
        all_cases,
        opponents=opponents,
        max_pairs=args.max_pairs,
    )
    exact_official_selection = (
        args.split == "test"
        and not args.development
        and policies == official_policies
        and opponents == tuple(config["opponents"])
        and args.max_pairs is None
        and args.max_decisions is None
        and args.ppo_checkpoint is None
        and args.bc_checkpoint is None
    )
    if args.split == "test" and not exact_official_selection:
        parser.error("Test evaluation must use the complete frozen official selection")

    commit, dirty = _git_provenance(root)
    if exact_official_selection and dirty:
        raise RuntimeError("Official M2 evaluation requires a clean tracked worktree")
    output_dir = _resolve(root, args.output)
    ensure_evidence_targets_absent(output_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    scenario_manifest = root / "assets/scenarios/crystal_run/manifest.json"
    scenario_hash = json.loads(scenario_manifest.read_text(encoding="utf-8"))[
        "wad_sha256"
    ]
    graph = RegionGraph.from_yaml(
        root / "assets/scenarios/crystal_run/src/regions.yaml"
    )
    config_cfg = root / "assets/scenarios/crystal_run/crystal_run.cfg"
    checkpoint_overrides = {
        "ppo": args.ppo_checkpoint,
        "bc": args.bc_checkpoint,
    }
    learned = {}
    checkpoint_hashes: dict[str, str] = {}
    for name in policies:
        if name not in ("ppo", "bc"):
            continue
        policy_spec = config["policies"][name]
        checkpoint = checkpoint_overrides[name] or _resolve(
            root, policy_spec["checkpoint"]
        )
        checkpoint = _resolve(root, checkpoint)
        digest = sha256_file(checkpoint)
        expected = policy_spec.get("expected_sha256")
        if expected is not None and digest != expected:
            raise ValueError(f"Frozen {name} checkpoint hash does not match")
        if exact_official_selection:
            selection_summary_path = _resolve(root, policy_spec["selection_summary"])
            selection_summary = json.loads(
                selection_summary_path.read_text(encoding="utf-8")
            )
            if selection_summary.get("selected_checkpoint_sha256") != digest:
                raise ValueError(
                    f"{name} checkpoint does not match its committed validation selection"
                )
        checkpoint_hashes[name] = digest
        learned[name] = load_actor_policy(
            name,
            checkpoint,
            device=device,
            expected_scenario_hash=scenario_hash,
        )

    records = []
    total = len(policies) * len(cases)
    completed = 0
    max_decisions = args.max_decisions or int(config["max_episode_decisions"])
    max_environment_attempts = int(config["max_environment_attempts"])
    for policy_name in policies:
        for case in cases:
            policy = learned.get(policy_name)
            if policy is None:
                policy = TeacherEvaluationPolicy(
                    policy_name, graph, side=case.learner_side
                )
            records.append(
                run_case_with_retries(
                    lambda case=case, policy=policy: run_m2_episode(
                        case,
                        policy=policy,
                        graph=graph,
                        config_path=config_cfg,
                        max_decisions=max_decisions,
                    ),
                    max_attempts=max_environment_attempts,
                )
            )
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"M2 evaluation progress: {completed}/{total}", flush=True)

    summary = evaluate_m2_records(
        records,
        official=exact_official_selection,
        expected_pairs_per_opponent=pairs_per_opponent,
        bootstrap_seed=int(config["bootstrap"]["seed"]),
        bootstrap_samples=int(config["bootstrap"]["samples"]),
        bootstrap_confidence=float(config["bootstrap"]["confidence"]),
        expected_scenario_hash=scenario_hash,
    )
    manifest = {
        "schema_version": 1,
        "official": exact_official_selection,
        "split": args.split,
        "git_commit": commit,
        "git_dirty": dirty,
        "config_sha256": sha256_file(config_path),
        "split_sha256": sha256_file(split_path),
        "scenario_manifest_sha256": sha256_file(scenario_manifest),
        "scenario_hash": scenario_hash,
        "checkpoint_sha256": checkpoint_hashes,
        "policies": list(policies),
        "opponents": list(opponents),
        "pairs_per_opponent": args.max_pairs or pairs_per_opponent,
        "max_episode_decisions": max_decisions,
        "max_environment_attempts": max_environment_attempts,
        "action_selection": config["action_selection"],
        "bootstrap": config["bootstrap"],
        "thresholds": config["thresholds"],
    }
    write_m2_evidence(
        output_dir,
        records=records,
        summary=summary,
        manifest=manifest,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True), flush=True)
    return 0 if args.development or summary.passed else 1
