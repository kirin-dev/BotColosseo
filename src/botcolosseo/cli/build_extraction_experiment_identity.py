from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.evaluation.extraction_identity import (
    build_experiment_identity,
    current_git_commit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the composite Extraction experiment identity"
    )
    parser.add_argument("--git-commit")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/extraction/showcase/experiment-identity.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment identity: {output}")
    payload = build_experiment_identity(
        root=root,
        git_commit=args.git_commit or current_git_commit(root),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
