from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from botcolosseo.agents.extraction_teachers import ExtractionStyle
from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.data.extraction_demonstrations import (
    ExtractionCase,
    load_extraction_cases,
)
from botcolosseo.evaluation.extraction import (
    evaluate_extraction_episode,
    summarize_extraction_episodes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an Extraction v2 policy on frozen paired cases"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    episodes = tuple(
        evaluate_extraction_episode(
            root=root,
            checkpoint=checkpoint,
            style=args.style,
            case=case,
            device=device,
        )
        for case in cases
    )
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
        "metrics": summarize_extraction_episodes(episodes),
        "scenario_hash": scenario_hash,
        "schema_version": 1,
        "split": args.split,
        "style": args.style,
        "test_cases_accessed": args.split == "test",
    }
    if args.output is not None:
        output = args.output if args.output.is_absolute() else root / args.output
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation: {output}")
        _atomic_json(result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
