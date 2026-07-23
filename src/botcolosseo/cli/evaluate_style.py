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
from botcolosseo.cli.train_ppo import _atomic_json
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_types import DuelActorObservation, DuelPrivilegedState
from botcolosseo.evaluation.m2 import load_duel_cases
from botcolosseo.evaluation.style import (
    STYLE_POLICIES,
    StyleEpisodeRecord,
    evaluate_aggressive_records,
    run_style_episode,
)
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS, DuelCase
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.training.bc import append_jsonl


class PublicStyleEvaluationPolicy:
    def __init__(self, name: str, policy: CheckpointOpponentPolicy) -> None:
        self.name = name
        self._policy = policy

    def reset(self, *, seed: int) -> None:
        del seed
        self._policy.reset()

    def act(self, observation: DuelActorObservation, state: DuelPrivilegedState) -> MacroAction:
        del state
        return MacroAction(self._policy.act(observation))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Base/Aggressive style retention on validation"
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--aggressive-checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("configs/m2/validation.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs-per-opponent", type=int, default=10)
    parser.add_argument("--max-decisions", type=int, default=525)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    return parser


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else root / path


def select_style_cases(
    cases: tuple[DuelCase, ...], *, pairs_per_opponent: int
) -> tuple[DuelCase, ...]:
    if not 0 < pairs_per_opponent <= 50:
        raise ValueError("M4 evaluation pairs per opponent must be in [1, 50]")
    selected = tuple(
        case
        for opponent in DUEL_OPPONENTS
        for case in [item for item in cases if item.opponent == opponent][: pairs_per_opponent * 2]
    )
    if len(selected) != len(DUEL_OPPONENTS) * pairs_per_opponent * 2:
        raise ValueError("M4 evaluation case source is incomplete")
    return selected


def _checkpoint_policy(
    name: str,
    path: Path,
    *,
    scenario_hash: str,
    device: torch.device,
) -> PublicStyleEvaluationPolicy:
    spec = OpponentSpec(
        opponent_id=name,
        kind="checkpoint",
        checkpoint=str(path),
        checkpoint_sha256=sha256_file(path),
        scenario_hash=scenario_hash,
        selection_evidence=f"m4-validation:{name}",
    )
    return PublicStyleEvaluationPolicy(name, CheckpointOpponentPolicy.load(spec, device=device))


def _record_from_dict(payload: dict[str, object]) -> StyleEpisodeRecord:
    names = {field.name for field in fields(StyleEpisodeRecord)}
    return StyleEpisodeRecord(**{name: payload[name] for name in names})  # type: ignore[arg-type]


def _load_records(path: Path) -> list[StyleEpisodeRecord]:
    if not path.exists():
        return []
    return [
        _record_from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_identity(
    *,
    base_checkpoint: Path,
    aggressive_checkpoint: Path,
    cases_path: Path,
    selected_cases: tuple[DuelCase, ...],
    scenario_hash: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage": "m4",
        "split": "validation",
        "base_checkpoint_sha256": sha256_file(base_checkpoint),
        "aggressive_checkpoint_sha256": sha256_file(aggressive_checkpoint),
        "cases_sha256": sha256_file(cases_path),
        "selected_case_ids": [
            [case.opponent, case.pair_index, case.learner_side] for case in selected_cases
        ],
        "scenario_hash": scenario_hash,
        "pairs_per_opponent": args.pairs_per_opponent,
        "max_decisions": args.max_decisions,
        "max_attempts": args.max_attempts,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "expected_episodes": len(STYLE_POLICIES) * len(selected_cases),
        "test_cases_accessed": False,
    }


def _run_with_retries(runner, *, max_attempts: int) -> StyleEpisodeRecord:
    for attempt in range(1, max_attempts + 1):
        try:
            return replace(runner(), environment_attempts=attempt)
        except RuntimeError as error:
            retriable = str(error) == ("Duel respawn did not complete within the warm-up limit")
            if not retriable or attempt == max_attempts:
                raise
    raise AssertionError("unreachable")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        min(
            args.pairs_per_opponent,
            args.max_decisions,
            args.max_attempts,
            args.bootstrap_samples,
        )
        <= 0
    ):
        raise ValueError("M4 evaluation settings must be positive")
    root = _project_root()
    base_checkpoint = _resolve(root, args.base_checkpoint)
    aggressive_checkpoint = _resolve(root, args.aggressive_checkpoint)
    cases_path = _resolve(root, args.cases)
    output_dir = _resolve(root, args.output_dir)
    for path in (base_checkpoint, aggressive_checkpoint, cases_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    scenario_hash = json.loads(
        (root / "assets/scenarios/crystal_run/manifest.json").read_text(encoding="utf-8")
    )["wad_sha256"]
    cases = load_duel_cases(cases_path, expected_split="validation", pairs_per_opponent=50)
    selected_cases = select_style_cases(cases, pairs_per_opponent=args.pairs_per_opponent)
    identity = _run_identity(
        base_checkpoint=base_checkpoint,
        aggressive_checkpoint=aggressive_checkpoint,
        cases_path=cases_path,
        selected_cases=selected_cases,
        scenario_hash=scenario_hash,
        args=args,
    )
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
            raise ValueError("M4 evaluation resume identity does not match")
    else:
        _atomic_json(identity, identity_path)
    episodes_path = output_dir / "episodes.jsonl"
    records = _load_records(episodes_path)
    completed = {
        (record.policy, record.opponent, record.pair_index, record.learner_side)
        for record in records
    }
    if len(completed) != len(records):
        raise ValueError("M4 evaluation episode ledger contains duplicates")
    graph = RegionGraph.from_yaml(root / "assets/scenarios/crystal_run/src/regions.yaml")
    config_path = root / "assets/scenarios/crystal_run/crystal_run.cfg"
    policies = {
        "strong_base": _checkpoint_policy(
            "strong_base",
            base_checkpoint,
            scenario_hash=scenario_hash,
            device=device,
        ),
        "aggressive": _checkpoint_policy(
            "aggressive",
            aggressive_checkpoint,
            scenario_hash=scenario_hash,
            device=device,
        ),
    }
    total = int(identity["expected_episodes"])
    for policy_name in STYLE_POLICIES:
        for case in selected_cases:
            key = (policy_name, case.opponent, case.pair_index, case.learner_side)
            if key in completed:
                continue
            record = _run_with_retries(
                lambda selected_case=case, selected_policy=policy_name: run_style_episode(
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
            print(f"M4 style evaluation progress: {len(records)}/{total}", flush=True)
    summary = evaluate_aggressive_records(
        records,
        expected_pairs_per_opponent=args.pairs_per_opponent,
        expected_scenario_hash=scenario_hash,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    summary_payload = {
        **summary.to_dict(),
        "schema_version": 1,
        "stage": "m4",
        "split": "validation",
        "checkpoint_sha256": {
            "strong_base": identity["base_checkpoint_sha256"],
            "aggressive": identity["aggressive_checkpoint_sha256"],
        },
        "test_cases_accessed": False,
    }
    _atomic_json(summary_payload, output_dir / "summary.json")
    manifest = {
        **identity,
        "episodes": len(records),
        "episodes_sha256": sha256_file(episodes_path),
        "summary_sha256": sha256_file(output_dir / "summary.json"),
        "passed": summary.passed,
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    print(json.dumps(summary_payload, indent=2, sort_keys=True))
    return 0 if summary.passed else 1
