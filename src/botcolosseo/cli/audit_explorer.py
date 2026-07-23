from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.evaluation.explorer_evidence_audit import audit_explorer_evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit M5 Explorer evidence")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    root = args.root or Path(__file__).resolve().parents[3]
    result = audit_explorer_evidence(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1
