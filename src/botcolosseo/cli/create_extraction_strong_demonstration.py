from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from botcolosseo.cli.select_extraction_candidate import (
    _atomic_json,
    _episodes,
    _load_report,
)
from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.evaluation.extraction_gates import strong_validation_gate

PRODUCT_MIN_EXTRACTION = 0.65


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an honestly scoped randomized Strong Showcase artifact"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--heldout-report", type=Path, required=True)
    parser.add_argument("--solo-report", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _rate(report: dict[str, object], attribute: str) -> float:
    episodes = _episodes(report)
    return sum(bool(getattr(item, attribute)) for item in episodes) / len(episodes)


def build_strong_demonstration(
    *,
    root: Path,
    checkpoint: Path,
    validation_path: Path,
    heldout_path: Path,
    solo_path: Path,
) -> dict[str, object]:
    validation = _load_report(validation_path, policy="strong", split="validation")
    heldout = _load_report(heldout_path, policy="strong", split="heldout")
    solo = _load_report(solo_path, policy="strong", split="solo")
    reports = (validation, heldout, solo)
    checkpoint_sha256 = sha256_file(checkpoint)
    if any(report.get("checkpoint_sha256") != checkpoint_sha256 for report in reports):
        raise ValueError("Strong demonstration checkpoint evidence does not match")
    if len({str(report.get("protocol_sha256")) for report in reports}) != 1:
        raise ValueError("Strong demonstration protocols do not match")
    if len({str(report.get("scenario_hash")) for report in reports}) != 1:
        raise ValueError("Strong demonstration scenarios do not match")
    if any(report.get("test_cases_accessed") is not False for report in reports):
        raise ValueError("Strong demonstration accessed official test cases")

    gate = strong_validation_gate(
        _episodes(validation), _episodes(heldout), _episodes(solo)
    )
    validation_extraction = _rate(validation, "extracted")
    heldout_extraction = _rate(heldout, "extracted")
    protocol_errors = sum(
        item.max_peer_tic_lag > 2 or item.truncated
        for report in reports
        for item in _episodes(report)
    )
    privilege_violations = sum(
        int(report.get("actor_privilege_violations", 0)) for report in reports
    )
    product_checks = [
        {
            "name": "validation_extraction",
            "passed": validation_extraction >= PRODUCT_MIN_EXTRACTION,
            "value": validation_extraction,
            "threshold": f">={PRODUCT_MIN_EXTRACTION:.2f}",
        },
        {
            "name": "heldout_extraction",
            "passed": heldout_extraction >= PRODUCT_MIN_EXTRACTION,
            "value": heldout_extraction,
            "threshold": f">={PRODUCT_MIN_EXTRACTION:.2f}",
        },
        {
            "name": "protocol_integrity",
            "passed": protocol_errors == 0,
            "value": float(protocol_errors),
            "threshold": "==0",
        },
        {
            "name": "actor_privilege_violations",
            "passed": privilege_violations == 0,
            "value": float(privilege_violations),
            "threshold": "==0",
        },
    ]
    failed_product_checks = [
        item["name"] for item in product_checks if not item["passed"]
    ]
    if failed_product_checks:
        raise ValueError(
            "Strong product demonstration failed checks: "
            + ", ".join(failed_product_checks)
        )

    evidence = (validation_path, heldout_path, solo_path)
    return {
        "schema_version": 1,
        "admission_schema_version": 2,
        "admission_kind": "strong_product_showcase",
        "admission_rule_timing": "post_validation_product_review",
        "claim_scope": "product_showcase_capability_only",
        "policy": "strong",
        "product_showcase_eligible": True,
        "research_gate_passed": gate.passed,
        "official_test_eligible": False,
        "research_failed_checks": [
            check.name for check in gate.checks if not check.passed
        ],
        "research_checks": [asdict(check) for check in gate.checks],
        "product_checks": product_checks,
        "candidate_checkpoint": str(checkpoint.relative_to(root)),
        "candidate_checkpoint_sha256": checkpoint_sha256,
        "evidence": [str(path.relative_to(root)) for path in evidence],
        "evidence_sha256": {
            str(path.relative_to(root)): sha256_file(path) for path in evidence
        },
        "protocol_sha256": validation["protocol_sha256"],
        "scenario_hash": validation["scenario_hash"],
        "actor_privilege_violations": privilege_violations,
        "test_cases_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    checkpoint = _resolve(root, args.checkpoint)
    output_checkpoint = _resolve(root, args.output_checkpoint)
    output_report = _resolve(root, args.output_report)
    if output_checkpoint.exists() or output_report.exists():
        raise FileExistsError("Refusing to overwrite a Strong demonstration")
    result = build_strong_demonstration(
        root=root,
        checkpoint=checkpoint,
        validation_path=_resolve(root, args.validation_report),
        heldout_path=_resolve(root, args.heldout_report),
        solo_path=_resolve(root, args.solo_report),
    )
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_checkpoint.with_name(f".{output_checkpoint.name}.tmp")
    shutil.copyfile(checkpoint, temporary)
    temporary.replace(output_checkpoint)
    result["showcase_checkpoint"] = str(output_checkpoint.relative_to(root))
    result["showcase_checkpoint_sha256"] = sha256_file(output_checkpoint)
    _atomic_json(result, output_report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
