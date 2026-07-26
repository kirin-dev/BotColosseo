from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.evaluation.extraction_release import audit_extraction_release


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the Crystal Run Extraction v2 release hashes"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/extraction-v2/showcase/manifest.json"),
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    report = args.report if args.report.is_absolute() else root / args.report
    payload = audit_extraction_release(report, root=root)
    print(
        json.dumps(
            {
                "artifacts": len(payload["artifacts"]),
                "showcase_acceptance": "PASS",
                "stage": payload["stage"],
                "test_cases_accessed": payload["test_cases_accessed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
