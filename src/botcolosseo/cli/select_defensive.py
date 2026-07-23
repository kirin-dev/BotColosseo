from __future__ import annotations

import argparse
import json
from functools import cmp_to_key
from pathlib import Path

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.cli.train_ppo import _atomic_json

_GRID = (0.25, 0.50, 0.75)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select one Defensive checkpoint from the frozen alpha grid"
    )
    parser.add_argument("--interpolation-dir", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _tag(alpha: float) -> str:
    return f"alpha-{round(alpha * 100):03d}"


def _candidate(
    alpha: float, *, interpolation_dir: Path, smoke_dir: Path
) -> dict[str, object]:
    tag = _tag(alpha)
    checkpoint = interpolation_dir / f"{tag}.pt"
    report_path = interpolation_dir / f"{tag}.json"
    summary_path = smoke_dir / tag / "summary.json"
    manifest_path = smoke_dir / tag / "manifest.json"
    for path in (checkpoint, report_path, summary_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_hash = sha256_file(checkpoint)
    identity_valid = (
        report.get("alpha") == alpha
        and report.get("style") == "defensive"
        and report.get("checkpoint_sha256") == checkpoint_hash
        and report.get("test_cases_accessed") is False
        and summary.get("checkpoint_sha256", {}).get("defensive") == checkpoint_hash
        and summary.get("test_cases_accessed") is False
        and manifest.get("passed") == summary.get("passed")
        and manifest.get("episodes") == 20
        and manifest.get("test_cases_accessed") is False
    )
    return {
        "alpha": alpha,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "eligible": bool(identity_valid and summary.get("passed") is True),
        "identity_valid": identity_valid,
        "protective_presence_delta": float(
            summary.get("protective_presence_delta", 0.0)
        ),
        "skill_retention": float(summary.get("skill_retention", 0.0)),
        "smoke_manifest_sha256": sha256_file(manifest_path),
        "smoke_summary_sha256": sha256_file(summary_path),
    }


def _compare(left: dict[str, object], right: dict[str, object]) -> int:
    left_shift = float(left["protective_presence_delta"])
    right_shift = float(right["protective_presence_delta"])
    if abs(left_shift - right_shift) > 0.05:
        return -1 if left_shift > right_shift else 1
    left_retention = float(left["skill_retention"])
    right_retention = float(right["skill_retention"])
    if left_retention != right_retention:
        return -1 if left_retention > right_retention else 1
    return -1 if float(left["alpha"]) < float(right["alpha"]) else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    interpolation_dir = (
        args.interpolation_dir.expanduser().resolve()
        if args.interpolation_dir.is_absolute()
        else root / args.interpolation_dir
    )
    smoke_dir = (
        args.smoke_dir.expanduser().resolve()
        if args.smoke_dir.is_absolute()
        else root / args.smoke_dir
    )
    candidates = [
        _candidate(alpha, interpolation_dir=interpolation_dir, smoke_dir=smoke_dir)
        for alpha in _GRID
    ]
    eligible = sorted(
        (item for item in candidates if item["eligible"]),
        key=cmp_to_key(_compare),
    )
    payload = {
        "schema_version": 1,
        "stage": "m5-defensive-selection",
        "grid": list(_GRID),
        "candidates": candidates,
        "passed": bool(eligible),
        "selected": eligible[0] if eligible else None,
        "test_cases_accessed": False,
    }
    output = args.output.expanduser().resolve() if args.output.is_absolute() else root / args.output
    _atomic_json(payload, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if eligible else 1
