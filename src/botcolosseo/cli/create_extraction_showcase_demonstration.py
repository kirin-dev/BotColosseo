from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from botcolosseo.cli.admit_extraction_showcase import (
    _atomic_json,
    _relative,
    _resolve,
    _verify_report_checkpoint,
)
from botcolosseo.cli.select_extraction_candidate import _episodes, _load_report
from botcolosseo.cli.select_extraction_showcases import _representative
from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.evaluation.extraction_gates import (
    aggressive_showcase_direction_counts,
    style_heldout_gate,
    style_showcase_direction_counts,
    style_validation_gate,
)

STYLES = ("aggressive", "defensive", "explorer")
ALLOWED_VALIDATION_FAILURES = {"style_ci_lower"}
CASE_STUDY_VALIDATION_FAILURES = {
    "paired_task_retention",
    "style_paired_difference",
    "style_ci_lower",
    "style_ci_upper",
    "anti_hack_real_upgrade_conversion",
    "reward_hacking_counterexamples",
}
CASE_STUDY_CAPABILITY_FLOORS = {
    "paired_task_retention": 0.75,
    "extraction_rate_delta": -0.10,
    "mean_value_ratio": 0.85,
    "worst_opponent_retention": 0.0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a validation-grounded Extraction Showcase demonstration"
    )
    parser.add_argument("--policy", choices=STYLES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--strong-validation-report", type=Path, required=True)
    parser.add_argument("--heldout-report", type=Path, required=True)
    parser.add_argument("--strong-heldout-report", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def build_validation_demonstration(
    *,
    root: Path,
    checkpoint: Path,
    validation_path: Path,
    strong_validation_path: Path,
    heldout_path: Path,
    strong_heldout_path: Path,
    policy: str,
) -> dict[str, object]:
    if policy not in STYLES:
        raise ValueError("Validation demonstration policy is unsupported")
    validation = _load_report(validation_path, policy=policy, split="validation")
    strong_validation = _load_report(
        strong_validation_path, policy="strong", split="validation"
    )
    heldout = _load_report(heldout_path, policy=policy, split="heldout")
    strong_heldout = _load_report(
        strong_heldout_path, policy="strong", split="heldout"
    )
    reports = (validation, strong_validation, heldout, strong_heldout)
    for report in reports:
        _verify_report_checkpoint(root, report)

    checkpoint_sha256 = sha256_file(checkpoint)
    strong_sha256 = str(strong_validation["checkpoint_sha256"])
    if (
        validation["checkpoint_sha256"] != checkpoint_sha256
        or heldout["checkpoint_sha256"] != checkpoint_sha256
    ):
        raise ValueError("Validation demonstration checkpoint evidence does not match")
    if (
        strong_heldout["checkpoint_sha256"] != strong_sha256
        or validation.get("base_checkpoint_sha256") != strong_sha256
        or heldout.get("base_checkpoint_sha256") != strong_sha256
    ):
        raise ValueError("Validation demonstration Strong Base lineage does not match")
    if len({report["protocol_sha256"] for report in reports}) != 1:
        raise ValueError("Validation demonstration protocols do not match")
    if len({report["scenario_hash"] for report in reports}) != 1:
        raise ValueError("Validation demonstration scenarios do not match")

    strong_validation_episodes = _episodes(strong_validation)
    validation_episodes = _episodes(validation)
    validation_gate = style_validation_gate(
        style=policy,
        strong=strong_validation_episodes,
        styled=validation_episodes,
    )
    validation_failures = [
        check.name for check in validation_gate.checks if not check.passed
    ]
    direction_counts = (
        aggressive_showcase_direction_counts(
            strong_validation_episodes,
            validation_episodes,
        )
        if policy == "aggressive"
        else style_showcase_direction_counts(
            style=policy,
            strong=strong_validation_episodes,
            styled=validation_episodes,
        )
    )
    strong_by_case = {
        (item.seed, item.learner_side, item.opponent_style): item
        for item in strong_validation_episodes
    }
    representative_case_count = sum(
        _representative(
            policy,
            item,
            strong_by_case[(item.seed, item.learner_side, item.opponent_style)],
        )
        for item in validation_episodes
    )
    unsafe_validation_failures = sorted(
        set(validation_failures) - ALLOWED_VALIDATION_FAILURES
    )
    if unsafe_validation_failures:
        checks = {check.name: check for check in validation_gate.checks}
        unsupported = sorted(
            set(validation_failures) - CASE_STUDY_VALIDATION_FAILURES
        )
        capability_floor_failed = [
            name
            for name, floor in CASE_STUDY_CAPABILITY_FLOORS.items()
            if checks[name].value < floor
        ]
        if unsupported or capability_floor_failed or representative_case_count < 1:
            raise ValueError(
                "Representative case demonstration has unsafe validation "
                "capability or direction failures: "
                f"unsupported={unsupported}, floors={capability_floor_failed}, "
                f"representative_cases={representative_case_count}"
            )
        evidence_tier = "representative_case_demonstration"
        claim_scope = "representative_validation_cases_only"
    else:
        evidence_tier = "validation_demonstration"
        claim_scope = "validation_distribution"
    heldout_gate = style_heldout_gate(
        strong=_episodes(strong_heldout),
        styled=_episodes(heldout),
    )
    heldout_failures = [
        check.name for check in heldout_gate.checks if not check.passed
    ]
    evidence = (
        validation_path,
        strong_validation_path,
        heldout_path,
        strong_heldout_path,
    )
    return {
        "schema_version": 1,
        "demonstration_schema_version": 1,
        "evidence_tier": evidence_tier,
        "admission_rule_timing": "post_validation_product_case_review",
        "claim_scope": claim_scope,
        "representative_case_count": representative_case_count,
        "aggregate_style_gate_passed": validation_gate.passed,
        "policy": policy,
        "product_demo_eligible": True,
        "research_gate_passed": False,
        "official_test_eligible": False,
        "research_failed_checks": [
            *validation_failures,
            *heldout_failures,
        ],
        "validation_failed_checks": validation_failures,
        "heldout_gate_passed": heldout_gate.passed,
        "heldout_failed_checks": heldout_failures,
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
        raise FileExistsError(
            "Refusing to overwrite an Extraction Showcase demonstration"
        )
    checkpoint_temporary = output_checkpoint.with_name(
        f".{output_checkpoint.name}.tmp"
    )
    report_temporary = output_report.with_name(f".{output_report.name}.tmp")
    if checkpoint_temporary.exists() or report_temporary.exists():
        raise FileExistsError("Refusing to overwrite stale Showcase temporary files")

    result = build_validation_demonstration(
        root=root,
        checkpoint=checkpoint,
        validation_path=_resolve(root, args.validation_report),
        strong_validation_path=_resolve(root, args.strong_validation_report),
        heldout_path=_resolve(root, args.heldout_report),
        strong_heldout_path=_resolve(root, args.strong_heldout_report),
        policy=args.policy,
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
