from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.evaluation.m6_release import build_m6_showcase_metric_payload
from botcolosseo.evaluation.showcase import canonical_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export frozen M6 showcase metrics")
    parser.add_argument("--m4", type=Path, required=True)
    parser.add_argument("--m4-showcase", type=Path, required=True)
    parser.add_argument("--defensive", type=Path, required=True)
    parser.add_argument("--explorer", type=Path, required=True)
    parser.add_argument("--difficulty", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    sources = {
        "m4": args.m4,
        "defensive": args.defensive,
        "explorer": args.explorer,
        "difficulty": args.difficulty,
    }
    for path in (*sources.values(), args.m4_showcase):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = build_m6_showcase_metric_payload(
        m4=_json(args.m4),
        m4_showcase=_json(args.m4_showcase),
        defensive=_json(args.defensive),
        explorer=_json(args.explorer),
        difficulty=_json(args.difficulty),
        upstream_sha256={
            name: sha256_file(path) for name, path in sources.items()
        },
    )
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite M6 showcase metrics")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload
