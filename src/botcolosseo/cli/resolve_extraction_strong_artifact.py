from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.data.demonstrations import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve a fail-closed Extraction Strong artifact"
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("runs/extraction/strong-ppo"),
    )
    parser.add_argument(
        "--field",
        choices=(
            "mode",
            "checkpoint",
            "manifest",
            "validation_report",
            "heldout_report",
            "solo_report",
        ),
    )
    return parser


def _verify_evidence(
    root: Path,
    payload: dict[str, object],
) -> dict[str, str]:
    evidence = payload.get("evidence")
    hashes = payload.get("evidence_sha256")
    if (
        not isinstance(evidence, list)
        or not isinstance(hashes, dict)
        or set(evidence) != set(hashes)
    ):
        raise ValueError("Strong artifact evidence manifest does not match")
    reports: dict[str, str] = {}
    for relative in evidence:
        path = root / str(relative)
        if not path.is_file() or sha256_file(path) != hashes[relative]:
            raise ValueError("Strong artifact evidence drifted")
        if not str(relative).endswith(".json"):
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        split = report.get("split")
        if (
            report.get("policy") == "strong"
            and split in {"validation", "heldout", "solo"}
        ):
            if split in reports:
                raise ValueError("Strong artifact has duplicate split evidence")
            if (
                report.get("metric_schema_version") != 2
                or report.get("complete") is not True
                or report.get("actor_privilege_violations") != 0
                or report.get("test_cases_accessed") is not False
            ):
                raise ValueError("Strong artifact report identity does not match")
            reports[str(split)] = str(relative)
    if set(reports) != {"validation", "heldout", "solo"}:
        raise ValueError("Strong artifact split evidence is incomplete")
    return reports


def resolve_strong_artifact(
    root: Path,
    directory: Path,
) -> dict[str, str]:
    directory = directory if directory.is_absolute() else root / directory
    selection_path = directory / "selection.json"
    selected_path = directory / "selected.pt"
    if selection_path.is_file() and selected_path.is_file():
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        expected = payload.get("selected_checkpoint_sha256")
        if (
            payload.get("gate_schema_version") == 2
            and payload.get("policy") == "strong"
            and payload.get("eligible") is True
            and payload.get("test_cases_accessed") is False
            and isinstance(expected, str)
            and sha256_file(selected_path) == expected
        ):
            reports = _verify_evidence(root, payload)
            return {
                "mode": "research_selection",
                "checkpoint": str(selected_path.relative_to(root)),
                "manifest": str(selection_path.relative_to(root)),
                **{f"{split}_report": path for split, path in reports.items()},
            }

    manifest_path = directory / "showcase-admission.json"
    checkpoint = directory / "showcase.pt"
    if not manifest_path.is_file() or not checkpoint.is_file():
        raise ValueError("No valid Strong research or product artifact exists")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = payload.get("showcase_checkpoint_sha256")
    checks = payload.get("research_checks")
    if not isinstance(checks, list):
        raise ValueError("Strong Showcase research checks are missing")
    failed = [
        check.get("name") for check in checks if check.get("passed") is not True
    ]
    heldout = next(
        (check for check in checks if check.get("name") == "heldout_extraction"),
        None,
    )
    if not (
        payload.get("admission_schema_version") == 1
        and payload.get("admission_kind") == "strong_product_showcase"
        and payload.get("admission_rule_timing")
        == "post_heldout_product_review"
        and payload.get("policy") == "strong"
        and payload.get("product_showcase_eligible") is True
        and payload.get("research_gate_passed") is False
        and payload.get("official_test_eligible") is False
        and payload.get("research_failed_checks") == ["heldout_extraction"]
        and failed == ["heldout_extraction"]
        and isinstance(heldout, dict)
        and heldout.get("value", 0) >= payload.get(
            "product_min_heldout_extraction", 1
        )
        and payload.get("actor_privilege_violations") == 0
        and payload.get("test_cases_accessed") is False
        and isinstance(expected, str)
        and sha256_file(checkpoint) == expected
    ):
        raise ValueError("Strong Showcase admission identity does not match")
    reports = _verify_evidence(root, payload)
    return {
        "mode": "product_showcase",
        "checkpoint": str(checkpoint.relative_to(root)),
        "manifest": str(manifest_path.relative_to(root)),
        **{f"{split}_report": path for split, path in reports.items()},
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    result = resolve_strong_artifact(root, args.directory)
    print(result[args.field] if args.field else json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
