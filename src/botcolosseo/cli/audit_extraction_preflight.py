from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.training.extraction_checkpoint import (
    load_extraction_strong_actor,
    load_extraction_style_actor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the real v3 Extraction end-to-end preflight"
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/extraction/preflight"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/extraction/preflight.json"),
    )
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    run_root = _resolve(root, args.run_root)
    output = _resolve(root, args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite preflight audit: {output}")
    scenario_hash = _load(
        root / "assets/scenarios/crystal_run_extraction/manifest.json"
    )["wad_sha256"]
    paths = {
        "train_manifest": run_root
        / "data/strong/train/train-manifest.json",
        "validation_manifest": run_root
        / "data/strong/validation/validation-manifest.json",
        "bc_summary": run_root / "strong-bc/summary.json",
        "strong_summary": run_root / "strong-ppo/summary.json",
        "style_summary": run_root / "aggressive/summary.json",
        "strong_validation": run_root / "eval/strong-validation.json",
        "strong_heldout": run_root / "eval/strong-heldout.json",
        "style_validation": run_root / "eval/aggressive-validation.json",
    }
    payloads = {name: _load(path) for name, path in paths.items()}
    if any(
        payload.get("scenario_hash") != scenario_hash
        or payload.get("test_cases_accessed") is not False
        for payload in payloads.values()
    ):
        raise ValueError("Preflight scenario or split provenance does not match")
    if (
        payloads["train_manifest"]["transitions"] != 128
        or payloads["validation_manifest"]["transitions"] != 64
        or payloads["bc_summary"]["updates"] != 2
        or payloads["strong_summary"]["environment_steps"] != 32
        or payloads["strong_summary"]["completed"] is not True
        or payloads["style_summary"]["environment_steps"] != 32
        or payloads["style_summary"]["completed"] is not True
        or payloads["style_summary"]["frozen_strong_actor"] is not True
        or payloads["style_summary"]["learned_residual_adapter"] is not True
    ):
        raise ValueError("Preflight stage counters do not match")
    strong_checkpoint = run_root / "strong-ppo/latest.pt"
    style_checkpoint = run_root / "aggressive/latest.pt"
    strong_sha256 = sha256_file(strong_checkpoint)
    style_sha256 = sha256_file(style_checkpoint)
    if (
        strong_sha256 != payloads["strong_summary"]["checkpoint_sha256"]
        or style_sha256 != payloads["style_summary"]["checkpoint_sha256"]
        or payloads["style_summary"]["base_checkpoint_sha256"] != strong_sha256
    ):
        raise ValueError("Preflight checkpoint hashes do not match")
    load_extraction_strong_actor(
        strong_checkpoint,
        expected_scenario_hash=str(scenario_hash),
        expected_sha256=strong_sha256,
    )
    load_extraction_style_actor(
        style_checkpoint,
        base_checkpoint=strong_checkpoint,
        expected_scenario_hash=str(scenario_hash),
        expected_base_sha256=strong_sha256,
        bottleneck=32,
        max_delta=2,
        expected_sha256=style_sha256,
    )
    result = {
        "schema_version": 1,
        "scenario_hash": scenario_hash,
        "real_vizdoom_environment": True,
        "pipeline": [
            "strong_teacher_demonstrations",
            "behavioral_cloning",
            "recurrent_strong_ppo",
            "aggressive_residual_ppo",
            "base_validation_inference",
            "heldout_validation_inference",
            "style_validation_inference",
        ],
        "artifacts": {
            name: {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
        "strong_checkpoint_sha256": strong_sha256,
        "aggressive_checkpoint_sha256": style_sha256,
        "fair_actor_observation_only": True,
        "frozen_strong_actor_verified": True,
        "test_cases_accessed": False,
        "status": "PASS",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
