from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.evaluation.extraction_official_test import (
    load_sealed_extraction_official_test,
)

POLICIES = ("strong", "aggressive", "defensive", "explorer")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the four selected Extraction policies before test"
    )
    for policy in POLICIES:
        parser.add_argument(
            f"--{policy}-selection",
            type=Path,
            required=True,
        )
    parser.add_argument(
        "--official-test-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/extraction/release/manifest.json"),
    )
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    output = _resolve(root, args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite release manifest: {output}")
    policies: dict[str, dict[str, object]] = {}
    protocol_hashes: set[str] = set()
    scenario_hashes: set[str] = set()
    strong_checkpoint_sha256 = ""
    for policy in POLICIES:
        selection_path = _resolve(root, getattr(args, f"{policy}_selection"))
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        check_names = {
            str(check.get("name"))
            for check in selection.get("checks", ())
            if check.get("passed") is True
        }
        required_checks = {
            "actor_privilege_violations",
            "protocol_integrity",
            "test_case_access",
        }
        if policy != "strong":
            required_checks.update(
                {
                    "anti_hack_no_timeout_value_loss"
                    if policy == "defensive"
                    else (
                        "anti_hack_complete_combat_chain"
                        if policy == "aggressive"
                        else "anti_hack_real_upgrade_conversion"
                    ),
                    "heldout_extraction_delta",
                    "reward_hacking_counterexamples",
                    "style_ci_lower",
                    "worst_opponent_retention",
                }
            )
        if (
            selection.get("policy") != policy
            or selection.get("gate_schema_version") != 2
            or selection.get("eligible") is not True
            or selection.get("test_cases_accessed") is not False
            or not required_checks.issubset(check_names)
        ):
            raise ValueError(f"{policy} selection is not release eligible")
        checkpoint = root / selection["selected_checkpoint"]
        checkpoint_sha256 = sha256_file(checkpoint)
        if checkpoint_sha256 != selection["selected_checkpoint_sha256"]:
            raise ValueError(f"{policy} selected checkpoint hash drifted")
        training_summary_path = checkpoint.parent / "summary.json"
        training_summary = json.loads(
            training_summary_path.read_text(encoding="utf-8")
        )
        if training_summary.get("test_cases_accessed") is not False:
            raise ValueError(f"{policy} training accessed test cases")
        base_checkpoint = training_summary.get("base_checkpoint")
        base_sha256 = training_summary.get("base_checkpoint_sha256")
        if policy == "strong":
            strong_checkpoint_sha256 = checkpoint_sha256
            base_checkpoint = None
            base_sha256 = None
        policies[policy] = {
            "checkpoint": str(checkpoint.relative_to(root)),
            "checkpoint_sha256": checkpoint_sha256,
            "selection_report": str(selection_path.relative_to(root)),
            "selection_report_sha256": sha256_file(selection_path),
            "training_summary": str(training_summary_path.relative_to(root)),
            "training_summary_sha256": sha256_file(training_summary_path),
            "base_checkpoint": base_checkpoint,
            "base_checkpoint_sha256": base_sha256,
        }
        protocol_hashes.add(selection["protocol_sha256"])
        scenario_hashes.add(selection["scenario_hash"])
    if len(protocol_hashes) != 1 or len(scenario_hashes) != 1:
        raise ValueError("Release policy evidence identities do not match")
    for policy in POLICIES[1:]:
        if policies[policy]["base_checkpoint_sha256"] != strong_checkpoint_sha256:
            raise ValueError(f"{policy} does not share the selected Strong Base")
    official_test_path = _resolve(root, args.official_test_manifest)
    official_test = load_sealed_extraction_official_test(official_test_path)
    protocol_sha256 = protocol_hashes.pop()
    scenario_hash = scenario_hashes.pop()
    if (
        official_test.validation_protocol_sha256 != protocol_sha256
        or official_test.scenario_hash != scenario_hash
    ):
        raise ValueError("Sealed official-test identities do not match the release")
    payload: dict[str, object] = {
        "schema_version": 1,
        "source_git_commit": _git_commit(root),
        "protocol_sha256": protocol_sha256,
        "scenario_hash": scenario_hash,
        "official_test_manifest": str(official_test_path.relative_to(root)),
        "official_test_manifest_sha256": sha256_file(official_test_path),
        "policies": policies,
        "test_cases_accessed": False,
        "test_cases_executed": False,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    payload["release_sha256"] = hashlib.sha256(canonical).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
