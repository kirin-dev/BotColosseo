from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.evaluation.difficulty_evidence_audit import (
    audit_difficulty_evidence,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit M5 difficulty evidence")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    root = args.root or Path(__file__).resolve().parents[3]
    result = audit_difficulty_evidence(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1
