from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.evaluation.extraction import ExtractionEpisodeMetrics
from botcolosseo.evaluation.extraction_gates import (
    strong_validation_gate,
    style_validation_gate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select an Extraction candidate using validation-only gates"
    )
    parser.add_argument(
        "--policy",
        choices=("strong", "aggressive", "defensive", "explorer"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--heldout-report", type=Path)
    parser.add_argument("--solo-report", type=Path)
    parser.add_argument("--strong-validation-report", type=Path)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _load_report(path: Path, *, policy: str, split: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("policy") != policy
        or payload.get("split") != split
        or payload.get("complete") is not True
        or payload.get("test_cases_accessed") is not False
    ):
        raise ValueError("Extraction evaluation report identity does not match")
    return payload


def _episodes(report: dict[str, object]) -> tuple[ExtractionEpisodeMetrics, ...]:
    return tuple(
        ExtractionEpisodeMetrics(**item)
        for item in report["metrics"]["episodes"]
    )


def _atomic_json(payload: object, path: Path) -> None:
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
    validation_path = _resolve(root, args.validation_report)
    output_checkpoint = _resolve(root, args.output_checkpoint)
    output_report = _resolve(root, args.output_report)
    if output_checkpoint.exists() or output_report.exists():
        raise FileExistsError("Refusing to overwrite an Extraction selection")
    validation = _load_report(
        validation_path,
        policy=args.policy,
        split="validation",
    )
    checkpoint_sha256 = sha256_file(checkpoint)
    if validation["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("Validation report checkpoint hash does not match")
    evidence = [validation_path]
    if args.policy == "strong":
        if (
            args.heldout_report is None
            or args.solo_report is None
            or args.strong_validation_report is not None
        ):
            raise ValueError("Strong selection requires heldout and solo reports")
        heldout_path = _resolve(root, args.heldout_report)
        solo_path = _resolve(root, args.solo_report)
        heldout = _load_report(
            heldout_path,
            policy="strong",
            split="heldout",
        )
        solo = _load_report(
            solo_path,
            policy="strong",
            split="solo",
        )
        if (
            heldout["checkpoint_sha256"] != checkpoint_sha256
            or solo["checkpoint_sha256"] != checkpoint_sha256
            or heldout["protocol_sha256"] != validation["protocol_sha256"]
            or solo["protocol_sha256"] != validation["protocol_sha256"]
        ):
            raise ValueError("Strong capability evidence identity does not match")
        gate = strong_validation_gate(
            _episodes(validation),
            _episodes(heldout),
            _episodes(solo),
        )
        evidence.extend((heldout_path, solo_path))
    else:
        if (
            args.strong_validation_report is None
            or args.heldout_report is not None
            or args.solo_report is not None
        ):
            raise ValueError("Style selection requires paired Strong validation")
        strong_path = _resolve(root, args.strong_validation_report)
        strong = _load_report(
            strong_path,
            policy="strong",
            split="validation",
        )
        if strong["protocol_sha256"] != validation["protocol_sha256"]:
            raise ValueError("Style paired evidence protocol does not match")
        gate = style_validation_gate(
            style=args.policy,
            strong=_episodes(strong),
            styled=_episodes(validation),
        )
        evidence.append(strong_path)
    result = {
        "schema_version": 1,
        "policy": args.policy,
        "eligible": gate.passed,
        "checks": [asdict(item) for item in gate.checks],
        "candidate_checkpoint": str(checkpoint.relative_to(root)),
        "candidate_checkpoint_sha256": checkpoint_sha256,
        "evidence": [str(path.relative_to(root)) for path in evidence],
        "evidence_sha256": {
            str(path.relative_to(root)): sha256_file(path) for path in evidence
        },
        "protocol_sha256": validation["protocol_sha256"],
        "scenario_hash": validation["scenario_hash"],
        "test_cases_accessed": False,
    }
    if not gate.passed:
        _atomic_json(result, output_report)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_checkpoint.with_name(f".{output_checkpoint.name}.tmp")
    shutil.copyfile(checkpoint, temporary)
    temporary.replace(output_checkpoint)
    result["selected_checkpoint"] = str(output_checkpoint.relative_to(root))
    result["selected_checkpoint_sha256"] = sha256_file(output_checkpoint)
    _atomic_json(result, output_report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
