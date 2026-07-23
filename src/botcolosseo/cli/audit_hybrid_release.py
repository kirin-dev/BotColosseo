from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.evaluation.hybrid_release import audit_hybrid_release


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit a portable learned/hybrid GitHub release package"
    )
    parser.add_argument("--package-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = audit_hybrid_release(args.package_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0
