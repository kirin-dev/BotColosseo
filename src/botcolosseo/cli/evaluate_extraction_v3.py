from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.evaluation.extraction import (
    evaluate_extraction_episode_with_retries,
    summarize_extraction_episodes,
)
from botcolosseo.evaluation.extraction_protocol import (
    balanced_extraction_case_subset,
    load_extraction_evaluation_protocol,
)
from botcolosseo.training.extraction_checkpoint import (
    load_extraction_strong_actor,
    load_extraction_style_actor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a v3 Extraction candidate without test access"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("strong", "aggressive", "defensive", "explorer"),
        required=True,
    )
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/extraction/evaluation.yaml"),
    )
    parser.add_argument(
        "--split",
        choices=("validation", "heldout", "solo"),
        required=True,
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--max-pairs-per-opponent", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _atomic_json(payload: object, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _training_summary(checkpoint: Path) -> dict[str, object]:
    path = checkpoint.parent / "summary.json"
    if not path.is_file():
        raise FileNotFoundError("Candidate training summary is missing")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("test_cases_accessed") is not False:
        raise ValueError("Candidate training summary accessed test cases")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_cases is not None and args.max_cases <= 0:
        raise ValueError("--max-cases must be positive")
    if (
        args.max_pairs_per_opponent is not None
        and args.max_pairs_per_opponent <= 0
    ):
        raise ValueError("--max-pairs-per-opponent must be positive")
    if args.max_cases is not None and args.max_pairs_per_opponent is not None:
        raise ValueError("Extraction evaluation subset options are mutually exclusive")
    root = Path(__file__).resolve().parents[3]
    checkpoint = _resolve(root, args.checkpoint)
    protocol_path = _resolve(root, args.protocol)
    output = _resolve(root, args.output)
    scenario_hash = json.loads(
        (
            root / "assets/scenarios/crystal_run_extraction/manifest.json"
        ).read_text(encoding="utf-8")
    )["wad_sha256"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint_sha256 = sha256_file(checkpoint)
    training_summary = _training_summary(checkpoint)
    if args.policy == "strong":
        if args.base_checkpoint is not None:
            raise ValueError("Strong evaluation does not accept --base-checkpoint")
        model, _ = load_extraction_strong_actor(
            checkpoint,
            expected_scenario_hash=scenario_hash,
            expected_sha256=checkpoint_sha256,
            device=device,
        )
        base_checkpoint = None
        base_sha256 = None
    else:
        if args.base_checkpoint is None:
            raise ValueError("Style evaluation requires --base-checkpoint")
        base_checkpoint = _resolve(root, args.base_checkpoint)
        base_sha256 = sha256_file(base_checkpoint)
        if (
            training_summary.get("style") != args.policy
            or training_summary.get("base_checkpoint_sha256") != base_sha256
            or training_summary.get("frozen_strong_actor") is not True
            or training_summary.get("frozen_strong_base") is not True
            or training_summary.get("learned_residual_adapter") is not True
        ):
            raise ValueError("Style training summary identity does not match")
        model, _ = load_extraction_style_actor(
            checkpoint,
            base_checkpoint=base_checkpoint,
            expected_scenario_hash=scenario_hash,
            expected_base_sha256=base_sha256,
            bottleneck=32,
            max_delta=2.0,
            expected_sha256=checkpoint_sha256,
            device=device,
            defensive_guardrail=args.policy == "defensive",
        )
    protocol = load_extraction_evaluation_protocol(protocol_path)
    cases = protocol.cases(args.split)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    elif args.max_pairs_per_opponent is not None:
        cases = balanced_extraction_case_subset(
            cases,
            pairs_per_opponent=args.max_pairs_per_opponent,
        )
    episodes = tuple(
        evaluate_extraction_episode_with_retries(
            root=root,
            checkpoint=checkpoint,
            style=args.policy,
            case=case,
            device=device,
            policy_model=model,
        )
        for case in cases
    )
    result = {
        "schema_version": 1,
        "metric_schema_version": 2,
        "disengagement_metric_version": 3,
        "policy": args.policy,
        "policy_kind": (
            "strong-recurrent-ppo"
            if args.policy == "strong"
            else (
                "learned-bounded-residual-with-risk-guardrail"
                if args.policy == "defensive"
                else "learned-bounded-residual"
            )
        ),
        "inference_guardrail": (
            "block_attack_when_low_resource_and_carried_value_ge_25"
            if args.policy == "defensive"
            else None
        ),
        "split": args.split,
        "complete": len(episodes)
        == protocol.splits[args.split].episode_count,
        "episodes_evaluated": len(episodes),
        "expected_episodes": protocol.splits[args.split].episode_count,
        "checkpoint": str(checkpoint.relative_to(root)),
        "checkpoint_sha256": checkpoint_sha256,
        "base_checkpoint": (
            str(base_checkpoint.relative_to(root))
            if base_checkpoint is not None
            else None
        ),
        "base_checkpoint_sha256": base_sha256,
        "protocol": str(protocol_path.relative_to(root)),
        "protocol_sha256": protocol.sha256,
        "scenario_hash": scenario_hash,
        "metrics": summarize_extraction_episodes(episodes),
        "actor_privilege_violations": 0,
        "fair_actor_observation_only": True,
        "test_cases_accessed": False,
    }
    _atomic_json(result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
