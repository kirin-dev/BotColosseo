from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from botcolosseo.cli.train_ppo import _atomic_json
from botcolosseo.evaluation.strong_base_selection import (
    load_candidate_evidence,
    select_strong_base,
)


def _resolve(root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select the M3 Strong Base from validation-only evidence"
    )
    parser.add_argument("--candidate-report", type=Path, action="append", required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[3]
    artifact_root = _resolve(project_root, args.artifact_root or Path("."))
    report_paths = tuple(
        sorted(
            (_resolve(artifact_root, path) for path in args.candidate_report),
            key=lambda path: str(path),
        )
    )
    candidates = tuple(
        load_candidate_evidence(path, artifact_root=artifact_root)
        for path in report_paths
    )
    decision = select_strong_base(candidates)
    output = _resolve(artifact_root, args.output)
    if output.exists():
        raise FileExistsError(f"Strong Base selection already exists: {output}")
    payload = {
        "schema_version": 1,
        "split": "validation",
        "test_cases_accessed": False,
        "selection_rule": list(decision.selection_rule),
        "candidate_reports": [
            path.relative_to(artifact_root).as_posix() for path in report_paths
        ],
        "candidates": list(decision.candidates),
        "selected": {
            **asdict(decision.selected),
            "rejection_reasons": list(decision.selected.rejection_reasons),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(payload, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
