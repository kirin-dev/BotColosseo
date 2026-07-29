from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from botcolosseo.cli.select_extraction_candidate import _episodes, _load_report
from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.evaluation.extraction_gates import (
    aggressive_showcase_direction_counts,
    style_heldout_gate,
    style_validation_gate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Admit directional Aggressive evidence for product Showcase use"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--strong-validation-report", type=Path, required=True)
    parser.add_argument("--heldout-report", type=Path, required=True)
    parser.add_argument("--strong-heldout-report", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite stale temporary file: {temporary}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _verify_report_checkpoint(root: Path, report: dict[str, object]) -> None:
    reported_path = report.get("checkpoint")
    if not isinstance(reported_path, str):
        raise ValueError("Extraction Showcase report has no checkpoint path")
    checkpoint = _resolve(root, Path(reported_path))
    if (
        not checkpoint.is_file()
        or sha256_file(checkpoint) != report["checkpoint_sha256"]
    ):
        raise ValueError("Extraction Showcase report checkpoint drifted")


def build_directional_showcase_admission(
    *,
    root: Path,
    checkpoint: Path,
    validation_path: Path,
    strong_validation_path: Path,
    heldout_path: Path,
    strong_heldout_path: Path,
) -> dict[str, object]:
    validation = _load_report(
        validation_path, policy="aggressive", split="validation"
    )
    strong_validation = _load_report(
        strong_validation_path, policy="strong", split="validation"
    )
    heldout = _load_report(heldout_path, policy="aggressive", split="heldout")
    strong_heldout = _load_report(
        strong_heldout_path, policy="strong", split="heldout"
    )
    for report in (validation, strong_validation, heldout, strong_heldout):
        _verify_report_checkpoint(root, report)

    checkpoint_sha256 = sha256_file(checkpoint)
    strong_sha256 = str(strong_validation["checkpoint_sha256"])
    reports = (validation, strong_validation, heldout, strong_heldout)
    if (
        validation["checkpoint_sha256"] != checkpoint_sha256
        or heldout["checkpoint_sha256"] != checkpoint_sha256
    ):
        raise ValueError("Aggressive Showcase checkpoint evidence does not match")
    if strong_heldout["checkpoint_sha256"] != strong_sha256:
        raise ValueError("Strong Showcase checkpoint evidence does not match")
    if (
        validation.get("base_checkpoint_sha256") != strong_sha256
        or heldout.get("base_checkpoint_sha256") != strong_sha256
    ):
        raise ValueError("Aggressive Showcase Strong Base lineage does not match")
    if len({report["protocol_sha256"] for report in reports}) != 1:
        raise ValueError("Aggressive Showcase protocols do not match")
    if len({report["scenario_hash"] for report in reports}) != 1:
        raise ValueError("Aggressive Showcase scenarios do not match")

    strong_validation_episodes = _episodes(strong_validation)
    validation_episodes = _episodes(validation)
    validation_gate = style_validation_gate(
        style="aggressive",
        strong=strong_validation_episodes,
        styled=validation_episodes,
    )
    failed_checks = [
        check.name for check in validation_gate.checks if not check.passed
    ]
    if failed_checks != ["style_ci_lower"]:
        raise ValueError(
            "Directional Showcase requires style_ci_lower as the sole research failure"
        )
    direction_counts = aggressive_showcase_direction_counts(
        strong_validation_episodes,
        validation_episodes,
    )
    if direction_counts["positive_pairs"] <= direction_counts["negative_pairs"]:
        raise ValueError("Aggressive Showcase has no positive paired majority")
    if (
        direction_counts["new_complete_chains"]
        <= direction_counts["lost_complete_chains"]
    ):
        raise ValueError("Aggressive Showcase has no positive complete-chain balance")

    heldout_gate = style_heldout_gate(
        strong=_episodes(strong_heldout),
        styled=_episodes(heldout),
    )
    if not heldout_gate.passed:
        raise ValueError("Aggressive Showcase heldout capability gate failed")

    evidence = (
        validation_path,
        strong_validation_path,
        heldout_path,
        strong_heldout_path,
    )
    return {
        "schema_version": 1,
        "admission_schema_version": 1,
        "admission_kind": "directional_showcase",
        "policy": "aggressive",
        "showcase_eligible": True,
        "research_gate_passed": False,
        "research_failed_checks": failed_checks,
        "validation_checks": [asdict(check) for check in validation_gate.checks],
        "heldout_checks": [asdict(check) for check in heldout_gate.checks],
        "direction_counts": direction_counts,
        "candidate_checkpoint": _relative(root, checkpoint),
        "candidate_checkpoint_sha256": checkpoint_sha256,
        "base_checkpoint_sha256": strong_sha256,
        "evidence": [_relative(root, path) for path in evidence],
        "evidence_sha256": {
            _relative(root, path): sha256_file(path) for path in evidence
        },
        "protocol_sha256": validation["protocol_sha256"],
        "scenario_hash": validation["scenario_hash"],
        "actor_privilege_violations": 0,
        "test_cases_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    checkpoint = _resolve(root, args.checkpoint)
    output_checkpoint = _resolve(root, args.output_checkpoint)
    output_report = _resolve(root, args.output_report)
    if output_checkpoint.exists() or output_report.exists():
        raise FileExistsError("Refusing to overwrite an Extraction Showcase admission")
    checkpoint_temporary = output_checkpoint.with_name(
        f".{output_checkpoint.name}.tmp"
    )
    report_temporary = output_report.with_name(f".{output_report.name}.tmp")
    if checkpoint_temporary.exists() or report_temporary.exists():
        raise FileExistsError("Refusing to overwrite stale Showcase temporary files")

    result = build_directional_showcase_admission(
        root=root,
        checkpoint=checkpoint,
        validation_path=_resolve(root, args.validation_report),
        strong_validation_path=_resolve(root, args.strong_validation_report),
        heldout_path=_resolve(root, args.heldout_report),
        strong_heldout_path=_resolve(root, args.strong_heldout_report),
    )
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint, checkpoint_temporary)
    checkpoint_temporary.replace(output_checkpoint)
    result["showcase_checkpoint"] = _relative(root, output_checkpoint)
    result["showcase_checkpoint_sha256"] = sha256_file(output_checkpoint)
    _atomic_json(result, output_report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
