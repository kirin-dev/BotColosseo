from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from botcolosseo.evaluation.m2_evidence_audit import (
    audit_official_evidence,
    audit_repository_provenance,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit official M2 release evidence")
    parser.add_argument("--report-dir", type=Path, default=Path("reports/m2"))
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    report_dir = (
        args.report_dir if args.report_dir.is_absolute() else root / args.report_dir
    )
    targets = tuple(
        report_dir / name
        for name in ("episodes.csv", "summary.json", "manifest.json")
    )
    existing = tuple(path.exists() for path in targets)
    if not any(existing) and args.allow_pending:
        print(json.dumps({"official_status": "pending", "passed": False}, indent=2))
        return 0
    if not all(existing):
        raise FileNotFoundError("Official M2 evidence is missing or partially written")
    result = audit_official_evidence(report_dir)
    result = audit_repository_provenance(root, report_dir, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
