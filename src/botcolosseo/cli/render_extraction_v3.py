from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.demo.extraction_showcase import record_extraction_showcase
from botcolosseo.envs.video import write_mp4
from botcolosseo.evaluation.extraction_protocol import (
    load_extraction_evaluation_protocol,
)
from botcolosseo.training.extraction_checkpoint import (
    load_extraction_strong_actor,
    load_extraction_style_actor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render one learned v3 Extraction validation replay"
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
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _atomic_json(payload: object, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite showcase evidence: {path}")
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
    checkpoint = _resolve(root, args.checkpoint)
    protocol_path = _resolve(root, args.protocol)
    output = _resolve(root, args.output)
    evidence = _resolve(root, args.evidence)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite showcase video: {output}")
    protocol = load_extraction_evaluation_protocol(protocol_path)
    cases = protocol.cases("validation")
    if not 0 <= args.case_index < len(cases):
        raise IndexError("Extraction showcase case index is outside protocol")
    scenario_hash = json.loads(
        (
            root / "assets/scenarios/crystal_run_extraction/manifest.json"
        ).read_text(encoding="utf-8")
    )["wad_sha256"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint_sha256 = sha256_file(checkpoint)
    if args.policy == "strong":
        if args.base_checkpoint is not None:
            raise ValueError("Strong showcase does not accept --base-checkpoint")
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
            raise ValueError("Style showcase requires --base-checkpoint")
        base_checkpoint = _resolve(root, args.base_checkpoint)
        base_sha256 = sha256_file(base_checkpoint)
        model, _ = load_extraction_style_actor(
            checkpoint,
            base_checkpoint=base_checkpoint,
            expected_scenario_hash=scenario_hash,
            expected_base_sha256=base_sha256,
            bottleneck=32,
            max_delta=2.0,
            expected_sha256=checkpoint_sha256,
            device=device,
        )
    episode = record_extraction_showcase(
        root=root,
        checkpoint=checkpoint,
        style=args.policy,
        case=cases[args.case_index],
        device=device,
        frame_stride=args.frame_stride,
        policy_model=model,
    )
    write_mp4(episode.frames, output, fps=args.fps)
    learner_side = episode.case.learner_side
    opponent_side = "opponent" if learner_side == "host" else "host"
    result = {
        **episode.record(),
        "schema_version": 1,
        "policy": args.policy,
        "policy_kind": (
            "strong-recurrent-ppo"
            if args.policy == "strong"
            else "learned-bounded-residual"
        ),
        "case_index": args.case_index,
        "protocol": str(protocol_path.relative_to(root)),
        "protocol_sha256": protocol.sha256,
        "checkpoint": str(checkpoint.relative_to(root)),
        "checkpoint_sha256": checkpoint_sha256,
        "base_checkpoint": (
            str(base_checkpoint.relative_to(root))
            if base_checkpoint is not None
            else None
        ),
        "base_checkpoint_sha256": base_sha256,
        "fps": args.fps,
        "frame_count": len(episode.frames),
        "frame_stride": args.frame_stride,
        "media": str(output.relative_to(root)),
        "media_sha256": sha256_file(output),
        "showcase_claims": {
            "valid_hits": episode.events.get(f"{learner_side}:valid_hit", 0),
            "kills": episode.events.get(f"{opponent_side}:death", 0),
            "cache_looted": episode.events.get(
                f"{learner_side}:cache_looted",
                0,
            ),
            "extracted": episode.extracted,
            "extracted_value": episode.extracted_value,
            "attack_decisions": episode.attack_decisions,
            "unique_route_cells": episode.unique_route_cells,
            "aggressive_chains": episode.aggressive_chains,
            "successful_disengagements": episode.successful_disengagements,
            "meaningful_extractions": episode.meaningful_extractions,
            "meaningful_loot_regions": episode.meaningful_loot_regions,
            "backpack_upgrades": episode.backpack_upgrades,
            "upgrade_to_extraction_conversions": (
                episode.upgrade_to_extraction_conversions
            ),
        },
        "test_cases_accessed": False,
    }
    _atomic_json(result, evidence)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
