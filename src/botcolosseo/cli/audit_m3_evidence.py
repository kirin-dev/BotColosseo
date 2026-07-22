from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from botcolosseo.evaluation.m3_evidence_audit import audit_m3_evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit official M3 evidence")
    parser.add_argument("--report-dir", type=Path, default=Path("reports/m3/official"))
    parser.add_argument(
        "--integrity-only",
        action="store_true",
        help="accept a complete integrity-clean result whose capability gate failed",
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    report_dir = (
        args.report_dir if args.report_dir.is_absolute() else root / args.report_dir
    )
    result = audit_m3_evidence(
        report_dir,
        artifact_root=root,
        require_capability_pass=not args.integrity_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
