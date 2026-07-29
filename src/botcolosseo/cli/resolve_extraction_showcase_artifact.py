from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.data.demonstrations import sha256_file

STYLES = ("aggressive", "defensive", "explorer")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve a fail-closed Extraction style Showcase artifact"
    )
    parser.add_argument("--policy", choices=STYLES, required=True)
    parser.add_argument("--directory", type=Path)
    return parser


def _verify_evidence(
    root: Path,
    *,
    policy: str,
    payload: dict[str, object],
) -> Path:
    evidence = payload.get("evidence")
    evidence_hashes = payload.get("evidence_sha256")
    if (
        not isinstance(evidence, list)
        or not isinstance(evidence_hashes, dict)
        or set(evidence) != set(evidence_hashes)
    ):
        raise ValueError("Style Showcase evidence manifest does not match")
    validation_reports: list[Path] = []
    for relative in evidence:
        path = root / str(relative)
        if not path.is_file() or sha256_file(path) != evidence_hashes[relative]:
            raise ValueError("Style Showcase evidence drifted")
        if str(relative).endswith("-validation.json"):
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("policy") == policy:
                validation_reports.append(path)
    if len(validation_reports) != 1:
        raise ValueError("Style Showcase has no unique validation report")
    report = json.loads(validation_reports[0].read_text(encoding="utf-8"))
    if (
        report.get("metric_schema_version") != 2
        or report.get("split") != "validation"
        or report.get("complete") is not True
        or report.get("fair_actor_observation_only") is not True
        or report.get("actor_privilege_violations") != 0
        or report.get("test_cases_accessed") is not False
    ):
        raise ValueError("Style Showcase validation report identity does not match")
    return validation_reports[0]


def _resolve_selection(
    root: Path,
    *,
    policy: str,
    directory: Path,
) -> dict[str, str] | None:
    manifest_path = directory / "selection.json"
    checkpoint = directory / "selected.pt"
    if not manifest_path.is_file() or not checkpoint.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = payload.get("selected_checkpoint_sha256")
    if not (
        payload.get("gate_schema_version") == 2
        and payload.get("policy") == policy
        and payload.get("eligible") is True
        and payload.get("test_cases_accessed") is False
        and isinstance(expected, str)
        and sha256_file(checkpoint) == expected
    ):
        return None
    validation = _verify_evidence(root, policy=policy, payload=payload)
    return {
        "mode": "research_selection",
        "checkpoint": str(checkpoint.relative_to(root)),
        "validation_report": str(validation.relative_to(root)),
        "manifest": str(manifest_path.relative_to(root)),
    }


def _validate_direction_counts(
    policy: str,
    counts: object,
) -> None:
    if not isinstance(counts, dict):
        raise ValueError("Style Showcase direction counts are missing")
    new_key, lost_key = (
        ("new_complete_chains", "lost_complete_chains")
        if policy == "aggressive"
        else ("new_showcase_chains", "lost_showcase_chains")
    )
    required = ("positive_pairs", "negative_pairs", new_key, lost_key)
    if (
        any(not isinstance(counts.get(key), int) for key in required)
        or counts["positive_pairs"] <= counts["negative_pairs"]
        or counts[new_key] <= counts[lost_key]
    ):
        raise ValueError("Style Showcase direction counts do not pass")


def _resolve_admission(
    root: Path,
    *,
    policy: str,
    directory: Path,
) -> dict[str, str]:
    manifest_path = directory / "showcase-admission.json"
    checkpoint = directory / "showcase.pt"
    if not manifest_path.is_file() or not checkpoint.is_file():
        raise ValueError(f"No valid {policy} Showcase artifact exists")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_timing = (
        "post_heldout_product_review"
        if policy == "aggressive"
        else "pre_heldout_product_rule"
    )
    validation_failures = payload.get("research_validation_failed_checks")
    heldout_failures = payload.get("original_heldout_failed_checks")
    original_checks = payload.get("original_heldout_checks")
    showcase_checks = payload.get("showcase_heldout_checks")
    if (
        not isinstance(original_checks, list)
        or not original_checks
        or not isinstance(showcase_checks, list)
        or not showcase_checks
    ):
        raise ValueError("Style Showcase heldout checks are missing")
    observed_heldout_failures = [
        check.get("name")
        for check in original_checks
        if check.get("passed") is not True
    ]
    expected = payload.get("showcase_checkpoint_sha256")
    if not (
        payload.get("admission_schema_version") == 1
        and payload.get("admission_kind") == "directional_showcase"
        and payload.get("admission_rule_timing") == expected_timing
        and payload.get("policy") == policy
        and payload.get("showcase_eligible") is True
        and payload.get("research_gate_passed") is False
        and validation_failures == ["style_ci_lower"]
        and heldout_failures == observed_heldout_failures
        and payload.get("research_failed_checks")
        == [*validation_failures, *heldout_failures]
        and payload.get("original_heldout_gate_passed")
        is (not heldout_failures)
        and all(check.get("passed") is True for check in showcase_checks)
        and payload.get("actor_privilege_violations") == 0
        and payload.get("test_cases_accessed") is False
        and isinstance(expected, str)
        and sha256_file(checkpoint) == expected
    ):
        raise ValueError("Style Showcase admission identity does not match")
    if any(
        name not in {"heldout_worst_opponent_retention"}
        for name in heldout_failures
    ):
        raise ValueError("Style Showcase original heldout evidence is unsafe")
    _validate_direction_counts(policy, payload.get("direction_counts"))
    validation = _verify_evidence(root, policy=policy, payload=payload)
    return {
        "mode": "directional_showcase",
        "checkpoint": str(checkpoint.relative_to(root)),
        "validation_report": str(validation.relative_to(root)),
        "manifest": str(manifest_path.relative_to(root)),
    }


def _resolve_demonstration(
    root: Path,
    *,
    policy: str,
    directory: Path,
) -> dict[str, str]:
    manifest_path = directory / "showcase-demonstration.json"
    checkpoint = directory / "showcase.pt"
    if not manifest_path.is_file() or not checkpoint.is_file():
        raise ValueError(f"No valid {policy} Showcase artifact exists")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_failures = payload.get("validation_failed_checks")
    heldout_failures = payload.get("heldout_failed_checks")
    heldout_checks = payload.get("heldout_checks")
    if (
        not isinstance(validation_failures, list)
        or any(name != "style_ci_lower" for name in validation_failures)
        or not isinstance(heldout_failures, list)
        or not isinstance(heldout_checks, list)
        or not heldout_checks
        or heldout_failures
        != [
            check.get("name")
            for check in heldout_checks
            if check.get("passed") is not True
        ]
    ):
        raise ValueError("Showcase demonstration check ledger does not match")
    expected = payload.get("showcase_checkpoint_sha256")
    if not (
        payload.get("demonstration_schema_version") == 1
        and payload.get("evidence_tier") == "validation_demonstration"
        and payload.get("policy") == policy
        and payload.get("product_demo_eligible") is True
        and payload.get("research_gate_passed") is False
        and payload.get("official_test_eligible") is False
        and payload.get("research_failed_checks")
        == [*validation_failures, *heldout_failures]
        and payload.get("heldout_gate_passed") is (not heldout_failures)
        and payload.get("actor_privilege_violations") == 0
        and payload.get("test_cases_accessed") is False
        and isinstance(expected, str)
        and sha256_file(checkpoint) == expected
    ):
        raise ValueError("Showcase demonstration identity does not match")
    validation = _verify_evidence(root, policy=policy, payload=payload)
    evidence = payload["evidence"]
    heldout_reports = []
    for relative in evidence:
        if not str(relative).endswith("-heldout.json"):
            continue
        report = json.loads((root / str(relative)).read_text(encoding="utf-8"))
        if report.get("policy") == policy:
            heldout_reports.append(report)
    if (
        len(heldout_reports) != 1
        or heldout_reports[0].get("split") != "heldout"
        or heldout_reports[0].get("complete") is not True
        or heldout_reports[0].get("test_cases_accessed") is not False
        or heldout_reports[0].get("actor_privilege_violations") != 0
    ):
        raise ValueError("Showcase demonstration heldout identity does not match")
    return {
        "mode": "validation_demonstration",
        "checkpoint": str(checkpoint.relative_to(root)),
        "validation_report": str(validation.relative_to(root)),
        "manifest": str(manifest_path.relative_to(root)),
    }


def resolve_style_showcase_artifacts(
    root: Path,
    *,
    policy: str,
    directory: Path,
) -> dict[str, str]:
    if policy not in STYLES:
        raise ValueError("Unsupported Extraction Showcase style")
    directory = directory if directory.is_absolute() else root / directory
    selection = _resolve_selection(
        root,
        policy=policy,
        directory=directory,
    )
    if selection is not None:
        return selection
    try:
        return _resolve_admission(
            root,
            policy=policy,
            directory=directory,
        )
    except ValueError as admission_error:
        try:
            return _resolve_demonstration(
                root,
                policy=policy,
                directory=directory,
            )
        except ValueError as demonstration_error:
            raise ValueError(
                f"No valid {policy} Showcase artifact exists: "
                f"{admission_error}; {demonstration_error}"
            ) from demonstration_error


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    directory = args.directory or Path(f"runs/extraction/styles/{args.policy}")
    result = resolve_style_showcase_artifacts(
        root,
        policy=args.policy,
        directory=directory,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
