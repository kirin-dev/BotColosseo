from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import torch

from botcolosseo.agents.difficulty import (
    DIFFICULTIES,
    DifficultyPolicy,
    load_difficulty_profiles,
)
from botcolosseo.agents.league_opponents import (
    CheckpointOpponentPolicy,
    OpponentSpec,
    sha256_file,
)
from botcolosseo.cli.evaluate_style import select_style_cases
from botcolosseo.cli.train_ppo import _atomic_json
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_types import DuelActorObservation, DuelPrivilegedState
from botcolosseo.evaluation.defensive import (
    PROTECTIVE_PRESENCE_ESTIMATOR,
    run_defensive_episode,
)
from botcolosseo.evaluation.explorer import (
    ROUTE_ENTROPY_ESTIMATOR,
    run_explorer_episode,
)
from botcolosseo.evaluation.m2 import load_duel_cases
from botcolosseo.evaluation.native_style_difficulty import (
    NativeEpisode,
    NativeStyle,
    NativeStyleDifficultyRecord,
    evaluate_native_style_difficulty,
)
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.training.bc import append_jsonl


class NativeDifficultyEvaluationPolicy:
    def __init__(
        self,
        name: str,
        policy: CheckpointOpponentPolicy,
        *,
        profile,
    ) -> None:
        self.name = name
        self._policy = DifficultyPolicy(policy, profile)

    def reset(self, *, seed: int) -> None:
        del seed
        self._policy.reset()

    def act(
        self,
        observation: DuelActorObservation,
        state: DuelPrivilegedState,
    ) -> MacroAction:
        del state
        return self._policy.act(observation)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Defensive or Explorer across difficulty profiles"
    )
    parser.add_argument("--style", choices=("defensive", "explorer"), required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--style-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/difficulty.yaml"))
    parser.add_argument("--cases", type=Path, default=Path("configs/m2/validation.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs-per-opponent", type=int, default=10)
    parser.add_argument("--max-decisions", type=int, default=525)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(
        args.pairs_per_opponent,
        args.max_decisions,
        args.max_attempts,
        args.bootstrap_samples,
    ) <= 0:
        raise ValueError("Native-style difficulty settings must be positive")
    style: NativeStyle = args.style
    root = Path(__file__).resolve().parents[3]
    base_path = _resolve(root, args.base_checkpoint)
    style_path = _resolve(root, args.style_checkpoint)
    config_path = _resolve(root, args.config)
    cases_path = _resolve(root, args.cases)
    output_dir = _resolve(root, args.output_dir)
    for path in (base_path, style_path, config_path, cases_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    profiles = load_difficulty_profiles(config_path)
    scenario_hash = json.loads(
        (root / "assets/scenarios/crystal_run/manifest.json").read_text(
            encoding="utf-8"
        )
    )["wad_sha256"]
    cases = load_duel_cases(
        cases_path,
        expected_split="validation",
        pairs_per_opponent=50,
    )
    selected = select_style_cases(
        cases,
        pairs_per_opponent=args.pairs_per_opponent,
    )
    checkpoint_hashes = {
        "strong_base": sha256_file(base_path),
        style: sha256_file(style_path),
    }
    estimator = (
        PROTECTIVE_PRESENCE_ESTIMATOR
        if style == "defensive"
        else ROUTE_ENTROPY_ESTIMATOR
    )
    policies = ("strong_base", style)
    identity = {
        "schema_version": 1,
        "stage": f"m5-{style}-difficulty",
        "style": style,
        "split": "validation",
        "checkpoint_sha256": checkpoint_hashes,
        "config_sha256": sha256_file(config_path),
        "cases_sha256": sha256_file(cases_path),
        "selected_case_ids": [
            [case.opponent, case.pair_index, case.learner_side]
            for case in selected
        ],
        "scenario_hash": scenario_hash,
        "estimator": estimator,
        "pairs_per_opponent": args.pairs_per_opponent,
        "max_decisions": args.max_decisions,
        "max_attempts": args.max_attempts,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "expected_episodes": len(policies)
        * len(DIFFICULTIES)
        * len(selected),
        "profiles": {
            name: {
                "reaction_delay": profile.reaction_delay,
                "policy_update_interval": profile.policy_update_interval,
            }
            for name, profile in profiles.items()
        },
        "test_cases_accessed": False,
    }
    if args.preflight:
        print(
            json.dumps(
                {**identity, "preflight_passed": True},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.json"
    if run_path.exists():
        if json.loads(run_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Native-style difficulty resume identity does not match")
    else:
        _atomic_json(identity, run_path)
    episodes_path = output_dir / "episodes.jsonl"
    records = _load_records(episodes_path, style=style)
    completed = {row.identity for row in records}
    if len(completed) != len(records):
        raise ValueError("Native-style difficulty ledger contains duplicates")
    templates = {
        "strong_base": _template(
            "strong_base",
            base_path,
            scenario_hash=scenario_hash,
            device=device,
            style=style,
        ),
        style: _template(
            style,
            style_path,
            scenario_hash=scenario_hash,
            device=device,
            style=style,
        ),
    }
    wrapped = {
        (policy, difficulty): NativeDifficultyEvaluationPolicy(
            policy,
            templates[policy].fork(),
            profile=profiles[difficulty],
        )
        for policy in policies
        for difficulty in DIFFICULTIES
    }
    graph = RegionGraph.from_yaml(
        root / "assets/scenarios/crystal_run/src/regions.yaml"
    )
    scenario_config = root / "assets/scenarios/crystal_run/crystal_run.cfg"
    total = int(identity["expected_episodes"])
    runner = _runner(style)
    for policy_name in policies:
        for difficulty in DIFFICULTIES:
            for case in selected:
                key = (
                    policy_name,
                    difficulty,
                    case.opponent,
                    case.pair_index,
                    case.learner_side,
                )
                if key in completed:
                    continue
                episode = _run_with_retries(
                    lambda selected_case=case,
                    selected_policy=policy_name,
                    selected_difficulty=difficulty: runner(
                        selected_case,
                        policy=wrapped[(selected_policy, selected_difficulty)],
                        graph=graph,
                        config_path=scenario_config,
                        max_decisions=args.max_decisions,
                    ),
                    max_attempts=args.max_attempts,
                )
                record = NativeStyleDifficultyRecord(difficulty, episode)
                append_jsonl(episodes_path, record.to_dict())
                records.append(record)
                completed.add(key)
                print(
                    f"M5 {style} difficulty progress: {len(records)}/{total}",
                    flush=True,
                )
    summary = evaluate_native_style_difficulty(
        records,
        style=style,
        expected_pairs_per_opponent=args.pairs_per_opponent,
        expected_scenario_hash=scenario_hash,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    payload = {
        **summary,
        "split": "validation",
        "checkpoint_sha256": checkpoint_hashes,
        "config_sha256": identity["config_sha256"],
        "scenario_hash": scenario_hash,
        "estimator": estimator,
        "test_cases_accessed": False,
    }
    summary_path = output_dir / "summary.json"
    _atomic_json(payload, summary_path)
    _atomic_json(
        {
            **identity,
            "episodes": len(records),
            "episodes_sha256": sha256_file(episodes_path),
            "summary_sha256": sha256_file(summary_path),
            "passed": summary["passed"],
        },
        output_dir / "manifest.json",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


def _resolve(root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else root / path


def _template(
    name: str,
    path: Path,
    *,
    scenario_hash: str,
    device: torch.device,
    style: NativeStyle,
) -> CheckpointOpponentPolicy:
    spec = OpponentSpec(
        opponent_id=name,
        kind="checkpoint",
        checkpoint=str(path),
        checkpoint_sha256=sha256_file(path),
        scenario_hash=scenario_hash,
        selection_evidence=f"m5-{style}-difficulty:{name}",
    )
    return CheckpointOpponentPolicy.load(spec, device=device)


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


def _runner(style: NativeStyle):
    if style == "defensive":
        return run_defensive_episode
    return run_explorer_episode
