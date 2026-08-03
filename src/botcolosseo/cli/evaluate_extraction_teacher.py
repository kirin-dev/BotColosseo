from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from botcolosseo.agents.extraction_teachers import (
    PrivilegedStrongExtractionTeacher,
)
from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.data.extraction_demonstrations import (
    extraction_teacher_sha256,
    load_extraction_cases,
)
from botcolosseo.envs.extraction_layouts import randomized_layout_variant
from botcolosseo.evaluation.extraction import (
    evaluate_extraction_episode_with_retries,
    summarize_extraction_episodes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the training-only mask-aware Strong Teacher"
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--scenario-directory",
        default="crystal_run_extraction_randomized",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _atomic_json(payload: object, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Teacher evaluation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_cases is not None and args.max_cases <= 0:
        raise ValueError("--max-cases must be positive")
    root = Path(__file__).resolve().parents[3]
    cases_path = args.cases if args.cases.is_absolute() else root / args.cases
    output = args.output if args.output.is_absolute() else root / args.output
    cases = load_extraction_cases(cases_path, expected_split="validation")
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    device = torch.device(args.device)
    episodes = tuple(
        evaluate_extraction_episode_with_retries(
            root=root,
            checkpoint=None,
            style="strong",
            case=case,
            device=device,
            privileged_teacher=PrivilegedStrongExtractionTeacher(
                side=case.learner_side,
                layout_variant=(
                    randomized_layout_variant(case.seed)
                    if case.layout_id == "randomized"
                    else None
                ),
            ),
            scenario_directory=args.scenario_directory,
        )
        for case in cases
    )
    result = {
        "schema_version": 1,
        "evaluation_kind": "training-only-privileged-teacher-ceiling",
        "case_manifest": str(cases_path.relative_to(root)),
        "case_manifest_sha256": sha256_file(cases_path),
        "teacher_implementation_sha256": extraction_teacher_sha256(),
        "scenario_directory": args.scenario_directory,
        "metrics": summarize_extraction_episodes(episodes),
        "protocol_inconsistencies": sum(
            episode.max_peer_tic_lag > 2 for episode in episodes
        ),
        "test_cases_accessed": False,
    }
    _atomic_json(result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
