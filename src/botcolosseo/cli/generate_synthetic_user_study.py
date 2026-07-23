from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.evaluation.synthetic_user_study import (
    generate_synthetic_user_study,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a clearly labelled synthetic M6 study preflight"
    )
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/m6/synthetic-user-study.json"),
    )
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args(argv)
    result = generate_synthetic_user_study(
        args.package_dir,
        args.config,
        responses_path=args.responses,
        provenance_path=args.provenance,
    )
    print(
        json.dumps(
            {
                "human_participants": False,
                "provenance": args.provenance.resolve().as_posix(),
                "responses": result["response_count"],
                "synthetic_data": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
