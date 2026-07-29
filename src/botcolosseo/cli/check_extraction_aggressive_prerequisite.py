from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.data.demonstrations import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check the fail-closed Aggressive downstream prerequisite"
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("runs/extraction/styles/aggressive"),
    )
    return parser


def check_aggressive_prerequisite(root: Path, directory: Path) -> str:
    directory = directory if directory.is_absolute() else root / directory
    selection_path = directory / "selection.json"
    selected_path = directory / "selected.pt"
    if selection_path.is_file() and selected_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        expected = selection.get("selected_checkpoint_sha256")
        if (
            selection.get("gate_schema_version") == 2
            and selection.get("policy") == "aggressive"
            and selection.get("eligible") is True
            and selection.get("test_cases_accessed") is False
            and isinstance(expected, str)
            and sha256_file(selected_path) == expected
        ):
            return "research_selection"

    admission_path = directory / "showcase-admission.json"
    showcase_path = directory / "showcase.pt"
    if not admission_path.is_file() or not showcase_path.is_file():
        raise ValueError("No valid Aggressive research selection or Showcase admission")
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    expected = admission.get("showcase_checkpoint_sha256")
    if not (
        admission.get("admission_schema_version") == 1
        and admission.get("admission_kind") == "directional_showcase"
        and admission.get("admission_rule_timing")
        == "post_heldout_product_review"
        and admission.get("policy") == "aggressive"
        and admission.get("showcase_eligible") is True
        and admission.get("research_gate_passed") is False
        and admission.get("research_failed_checks")
        == ["style_ci_lower", "heldout_worst_opponent_retention"]
        and admission.get("research_validation_failed_checks")
        == ["style_ci_lower"]
        and admission.get("original_heldout_gate_passed") is False
        and admission.get("original_heldout_failed_checks")
        == ["heldout_worst_opponent_retention"]
        and admission.get("actor_privilege_violations") == 0
        and admission.get("test_cases_accessed") is False
        and isinstance(expected, str)
        and sha256_file(showcase_path) == expected
    ):
        raise ValueError("Aggressive Showcase admission identity does not match")
    showcase_checks = admission.get("showcase_heldout_checks")
    if (
        not isinstance(showcase_checks, list)
        or not showcase_checks
        or any(check.get("passed") is not True for check in showcase_checks)
    ):
        raise ValueError("Aggressive Showcase heldout checks do not pass")
    evidence_hashes = admission.get("evidence_sha256")
    if not isinstance(evidence_hashes, dict) or not evidence_hashes:
        raise ValueError("Aggressive Showcase admission has no bound evidence")
    for relative, digest in evidence_hashes.items():
        path = root / str(relative)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError("Aggressive Showcase evidence drifted")
    return "directional_showcase"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    mode = check_aggressive_prerequisite(root, args.directory)
    print(mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
