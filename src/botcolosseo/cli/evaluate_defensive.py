from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace
from pathlib import Path

import torch

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
    DEFENSIVE_POLICIES,
    DefensiveEpisodeRecord,
    evaluate_defensive_records,
    run_defensive_episode,
)
from botcolosseo.evaluation.m2 import load_duel_cases
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.training.bc import append_jsonl


class PublicDefensiveEvaluationPolicy:
    def __init__(self, name: str, policy: CheckpointOpponentPolicy) -> None:
        self.name = name
        self._policy = policy

    def reset(self, *, seed: int) -> None:
        del seed
        self._policy.reset()

    def act(
        self, observation: DuelActorObservation, state: DuelPrivilegedState
    ) -> MacroAction:
        del state
        return MacroAction(self._policy.act(observation))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Base/Defensive style retention on validation"
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--defensive-checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("configs/m3/validation.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs-per-opponent", type=int, default=10)
    parser.add_argument("--max-decisions", type=int, default=525)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else root / path


def _checkpoint_policy(
    name: str, path: Path, *, scenario_hash: str, device: torch.device
) -> PublicDefensiveEvaluationPolicy:
    spec = OpponentSpec(
        opponent_id=name,
        kind="checkpoint",
        checkpoint=str(path),
        checkpoint_sha256=sha256_file(path),
        scenario_hash=scenario_hash,
        selection_evidence=f"m5-defensive-validation:{name}",
    )
    return PublicDefensiveEvaluationPolicy(
        name, CheckpointOpponentPolicy.load(spec, device=device)
    )


def _load_records(path: Path) -> list[DefensiveEpisodeRecord]:
    if not path.exists():
        return []
    names = {field.name for field in fields(DefensiveEpisodeRecord)}
    return [
        DefensiveEpisodeRecord(
            **{name: payload[name] for name in names}  # type: ignore[arg-type]
        )
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for payload in (json.loads(line),)
    ]


def _run_with_retries(runner, *, max_attempts: int) -> DefensiveEpisodeRecord:
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(
        args.pairs_per_opponent,
        args.max_decisions,
        args.max_attempts,
        args.bootstrap_samples,
    ) <= 0:
        raise ValueError("M5 Defensive evaluation settings must be positive")
    root = Path(__file__).resolve().parents[3]
    base_path = _resolve(root, args.base_checkpoint)
    defensive_path = _resolve(root, args.defensive_checkpoint)
    cases_path = _resolve(root, args.cases)
    output_dir = _resolve(root, args.output_dir)
    for path in (base_path, defensive_path, cases_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    scenario_hash = json.loads(
        (root / "assets/scenarios/crystal_run/manifest.json").read_text(encoding="utf-8")
    )["wad_sha256"]
    cases = load_duel_cases(
        cases_path, expected_split="validation", pairs_per_opponent=50
    )
    selected = select_style_cases(cases, pairs_per_opponent=args.pairs_per_opponent)
    identity = {
        "schema_version": 1,
        "stage": "m5-defensive",
        "split": "validation",
        "base_checkpoint_sha256": sha256_file(base_path),
        "defensive_checkpoint_sha256": sha256_file(defensive_path),
        "cases_sha256": sha256_file(cases_path),
        "selected_case_ids": [
            [case.opponent, case.pair_index, case.learner_side] for case in selected
        ],
        "scenario_hash": scenario_hash,
        "pairs_per_opponent": args.pairs_per_opponent,
        "max_decisions": args.max_decisions,
        "max_attempts": args.max_attempts,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "expected_episodes": len(DEFENSIVE_POLICIES) * len(selected),
        "test_cases_accessed": False,
    }
    if args.preflight:
        print(json.dumps({**identity, "preflight_passed": True}, indent=2, sort_keys=True))
        return 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "run.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("M5 Defensive evaluation resume identity does not match")
    else:
        _atomic_json(identity, identity_path)
    episodes_path = output_dir / "episodes.jsonl"
    records = _load_records(episodes_path)
    completed = {
        (row.policy, row.opponent, row.pair_index, row.learner_side) for row in records
    }
    if len(completed) != len(records):
        raise ValueError("M5 Defensive episode ledger contains duplicates")
    graph = RegionGraph.from_yaml(root / "assets/scenarios/crystal_run/src/regions.yaml")
    config_path = root / "assets/scenarios/crystal_run/crystal_run.cfg"
    policies = {
        "strong_base": _checkpoint_policy(
            "strong_base", base_path, scenario_hash=scenario_hash, device=device
        ),
        "defensive": _checkpoint_policy(
            "defensive", defensive_path, scenario_hash=scenario_hash, device=device
        ),
    }
    total = int(identity["expected_episodes"])
    for policy_name in DEFENSIVE_POLICIES:
        for case in selected:
            key = (policy_name, case.opponent, case.pair_index, case.learner_side)
            if key in completed:
                continue
            record = _run_with_retries(
                lambda selected_case=case, selected_policy=policy_name: run_defensive_episode(
                    selected_case,
                    policy=policies[selected_policy],
                    graph=graph,
                    config_path=config_path,
                    max_decisions=args.max_decisions,
                ),
                max_attempts=args.max_attempts,
            )
            append_jsonl(episodes_path, record.to_dict())
            records.append(record)
            completed.add(key)
            print(f"M5 Defensive evaluation progress: {len(records)}/{total}", flush=True)
    summary = evaluate_defensive_records(
        records,
        expected_pairs_per_opponent=args.pairs_per_opponent,
        expected_scenario_hash=scenario_hash,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    payload = {
        **summary.to_dict(),
        "schema_version": 1,
        "stage": "m5-defensive",
        "split": "validation",
        "checkpoint_sha256": {
            "strong_base": identity["base_checkpoint_sha256"],
            "defensive": identity["defensive_checkpoint_sha256"],
        },
        "test_cases_accessed": False,
    }
    _atomic_json(payload, output_dir / "summary.json")
    _atomic_json(
        {
            **identity,
            "episodes": len(records),
            "episodes_sha256": sha256_file(episodes_path),
            "summary_sha256": sha256_file(output_dir / "summary.json"),
            "passed": summary.passed,
        },
        output_dir / "manifest.json",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if summary.passed else 1
