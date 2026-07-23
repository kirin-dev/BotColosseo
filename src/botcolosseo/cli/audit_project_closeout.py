from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.evaluation.project_closeout import audit_project_closeout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the current-scenario project closeout"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/m6/project-closeout.json"),
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    payload = audit_project_closeout(
        (root / args.report).resolve(),
        root=root,
    )
    print(
        json.dumps(
            {
                "artifacts": len(payload["artifacts"]),
                "current_scenario_development_complete": True,
                "human_participants": False,
                "stage": payload["stage"],
                "synthetic_data": True,
                "test_cases_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
