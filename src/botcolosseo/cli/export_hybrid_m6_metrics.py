from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.evaluation.hybrid_m6_metrics import (
    build_hybrid_m6_metric_payload,
)
from botcolosseo.evaluation.showcase import canonical_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export hash-bound hybrid M6 product metrics"
    )
    parser.add_argument("--aggressive", type=Path, required=True)
    parser.add_argument("--defensive", type=Path, required=True)
    parser.add_argument("--explorer", type=Path, required=True)
    parser.add_argument("--difficulty", type=Path, required=True)
    parser.add_argument("--showcase", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    sources = {
        "aggressive": args.aggressive,
        "defensive": args.defensive,
        "explorer": args.explorer,
        "difficulty": args.difficulty,
        "showcase": args.showcase,
    }
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = build_hybrid_m6_metric_payload(
        aggressive=_json(args.aggressive),
        defensive=_json(args.defensive),
        explorer=_json(args.explorer),
        difficulty=_json(args.difficulty),
        showcase=_json(args.showcase),
        upstream_sha256={
            name: sha256_file(path) for name, path in sources.items()
        },
    )
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite hybrid M6 metrics")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload
