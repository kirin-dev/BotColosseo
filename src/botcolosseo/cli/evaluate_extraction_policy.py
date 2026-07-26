from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from botcolosseo.agents.extraction_governor import AggressiveCapabilityGovernor
from botcolosseo.agents.extraction_model import load_extraction_policy
from botcolosseo.agents.extraction_teachers import ExtractionStyle
from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.data.extraction_demonstrations import (
    ExtractionCase,
    load_extraction_cases,
)
from botcolosseo.evaluation.extraction import (
    evaluate_extraction_episode_with_retries,
    is_aggressive_showcase_chain,
    summarize_extraction_episodes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an Extraction v2 policy on frozen paired cases"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--aggressive-governor", action="store_true")
    parser.add_argument("--governor-carried", type=int, default=35)
    parser.add_argument("--governor-health", type=int, default=40)
    parser.add_argument("--governor-remaining", type=float, default=40.0)
    parser.add_argument(
        "--style",
        choices=tuple(style.value for style in ExtractionStyle),
        required=True,
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("configs/extraction_v2/validation.json"),
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--stop-on-aggressive-chain",
        action="store_true",
        help="Stop after a replay proves hit, kill, cache-loot, and extraction.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.split == "test" and not args.allow_test:
        raise ValueError("Test evaluation requires explicit --allow-test")
    if args.max_cases is not None and args.max_cases <= 0:
        raise ValueError("--max-cases must be positive")
    root = Path(__file__).resolve().parents[3]
    checkpoint = (
        args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    )
    cases_path = args.cases if args.cases.is_absolute() else root / args.cases
    if args.split == "test":
        payload = json.loads(cases_path.read_text(encoding="utf-8"))
        cases = tuple(ExtractionCase(**item) for item in payload["cases"])
    else:
        cases = load_extraction_cases(cases_path, expected_split=args.split)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    policy_model = None
    base_checkpoint = None
    if args.aggressive_governor:
        if args.style != ExtractionStyle.AGGRESSIVE.value:
            raise ValueError("Aggressive governor requires --style aggressive")
        if args.base_checkpoint is None:
            raise ValueError("Aggressive governor requires --base-checkpoint")
        base_checkpoint = (
            args.base_checkpoint
            if args.base_checkpoint.is_absolute()
            else root / args.base_checkpoint
        )
        scenario_hash = json.loads(
            (
                root / "assets/scenarios/crystal_run_extraction/manifest.json"
            ).read_text(encoding="utf-8")
        )["wad_sha256"]
        strong, _ = load_extraction_policy(
            base_checkpoint,
            style=ExtractionStyle.STRONG.value,
            scenario_hash=scenario_hash,
            device=device,
        )
        aggressive, _ = load_extraction_policy(
            checkpoint,
            style=args.style,
            scenario_hash=scenario_hash,
            device=device,
        )
        policy_model = AggressiveCapabilityGovernor(
            strong_base=strong,
            aggressive=aggressive,
            carried_value_threshold=args.governor_carried,
            health_threshold=args.governor_health,
            remaining_time_threshold=args.governor_remaining,
        ).to(device)
        policy_model.eval()
    if args.stop_on_aggressive_chain and args.style != ExtractionStyle.AGGRESSIVE.value:
        raise ValueError("Aggressive chain search requires --style aggressive")
    evaluated = []
    selected = None
    for case in cases:
        episode = evaluate_extraction_episode_with_retries(
            root=root,
            checkpoint=checkpoint,
            style=args.style,
            case=case,
            device=device,
            policy_model=policy_model,
        )
        evaluated.append(episode)
        if args.stop_on_aggressive_chain and is_aggressive_showcase_chain(episode):
            selected = episode
            break
    episodes = tuple(evaluated)
    scenario_hash = json.loads(
        (
            root / "assets/scenarios/crystal_run_extraction/manifest.json"
        ).read_text(encoding="utf-8")
    )["wad_sha256"]
    result = {
        "case_manifest": str(cases_path.relative_to(root)),
        "case_manifest_sha256": sha256_file(cases_path),
        "checkpoint": _display_path(checkpoint, root),
        "checkpoint_sha256": sha256_file(checkpoint),
        "base_checkpoint": (
            _display_path(base_checkpoint, root)
            if base_checkpoint is not None
            else None
        ),
        "base_checkpoint_sha256": (
            sha256_file(base_checkpoint) if base_checkpoint is not None else None
        ),
        "metrics": summarize_extraction_episodes(episodes),
        "scenario_hash": scenario_hash,
        "schema_version": 1,
        "split": args.split,
        "style": args.style,
        "policy_kind": (
            "public-observation-capability-governor"
            if args.aggressive_governor
            else "learned-checkpoint"
        ),
        "governor_thresholds": (
            {
                "carried_value": args.governor_carried,
                "health": args.governor_health,
                "remaining_time": args.governor_remaining,
            }
            if args.aggressive_governor
            else None
        ),
        "test_cases_accessed": args.split == "test",
    }
    if args.stop_on_aggressive_chain:
        result["search"] = {
            "cases_evaluated": len(episodes),
            "manifest_cases": len(cases),
            "selected_episode": (
                vars(selected) if selected is not None else None
            ),
            "stopped_early": selected is not None and len(episodes) < len(cases),
            "success": selected is not None,
        }
    if args.output is not None:
        output = args.output if args.output.is_absolute() else root / args.output
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation: {output}")
        _atomic_json(result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
