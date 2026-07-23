from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.evaluation.user_study_curation import (
    curate_user_study_clips,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the audited six-clip M6 curation"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/m6/user-study-curation.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    manifest = curate_user_study_clips(
        args.config,
        output_dir=args.output_dir,
        root=root,
    )
    print(
        json.dumps(
            {
                "clips": manifest["clip_count"],
                "output_dir": (root / args.output_dir).resolve().as_posix(),
                "test_cases_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
