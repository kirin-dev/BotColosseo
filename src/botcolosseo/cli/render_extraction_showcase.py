from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from botcolosseo.agents.extraction_teachers import ExtractionStyle
from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.data.extraction_demonstrations import load_extraction_cases
from botcolosseo.demo.extraction_showcase import record_extraction_showcase
from botcolosseo.envs.video import write_mp4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a real validation Extraction v2 policy replay"
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
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    return parser


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    checkpoint = (
        args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    )
    cases_path = args.cases if args.cases.is_absolute() else root / args.cases
    cases = load_extraction_cases(cases_path, expected_split="validation")
    if not 0 <= args.case_index < len(cases):
        raise IndexError("Extraction showcase case index is outside manifest")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    episode = record_extraction_showcase(
        root=root,
        checkpoint=checkpoint,
        style=args.style,
        case=cases[args.case_index],
        device=device,
        frame_stride=args.frame_stride,
    )
    output = args.output if args.output.is_absolute() else root / args.output
    write_mp4(episode.frames, output, fps=args.fps)
    result = {
        **episode.record(),
        "case_manifest": str(cases_path.relative_to(root)),
        "case_manifest_sha256": sha256_file(cases_path),
        "checkpoint_sha256": sha256_file(checkpoint),
        "fps": args.fps,
        "frame_count": len(episode.frames),
        "frame_stride": args.frame_stride,
        "media": str(output.relative_to(root)) if output.is_relative_to(root) else str(output),
        "media_sha256": sha256_file(output),
        "schema_version": 1,
    }
    if args.evidence is not None:
        evidence = (
            args.evidence if args.evidence.is_absolute() else root / args.evidence
        )
        if evidence.exists():
            raise FileExistsError(f"Refusing to overwrite evidence: {evidence}")
        _atomic_json(result, evidence)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
