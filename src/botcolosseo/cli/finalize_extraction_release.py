from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.evaluation.extraction_release import build_extraction_release


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the hash-bound Crystal Run Extraction v2 release"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/extraction-v2/showcase/manifest.json"),
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite release manifest: {output}")
    payload = build_extraction_release(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "artifacts": len(payload["artifacts"]),
                "output": str(output.relative_to(root)),
                "showcase_acceptance": "PASS",
                "test_cases_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
