from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.evaluation.hybrid_showcase import select_hybrid_showcase_case
from botcolosseo.evaluation.m2 import load_duel_cases
from botcolosseo.evaluation.showcase import canonical_json


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("Hybrid showcase sources must stay inside the repository") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select a validation case exposing all hybrid style mechanisms"
    )
    parser.add_argument("--aggressive-episodes", type=Path, required=True)
    parser.add_argument("--defensive-episodes", type=Path, required=True)
    parser.add_argument("--defensive-telemetry", type=Path, required=True)
    parser.add_argument("--explorer-episodes", type=Path, required=True)
    parser.add_argument("--explorer-telemetry", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("configs/m2/validation.json"))
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--selection-evidence", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    inputs = {
        "aggressive_episodes": (root / args.aggressive_episodes).resolve(),
        "defensive_episodes": (root / args.defensive_episodes).resolve(),
        "defensive_telemetry": (root / args.defensive_telemetry).resolve(),
        "explorer_episodes": (root / args.explorer_episodes).resolve(),
        "explorer_telemetry": (root / args.explorer_telemetry).resolve(),
    }
    cases_path = (root / args.cases).resolve()
    output_manifest = (root / args.output_manifest).resolve()
    selection_path = (root / args.selection_evidence).resolve()
    for path in (*inputs.values(), cases_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_manifest.exists() or selection_path.exists():
        raise FileExistsError("Refusing to overwrite hybrid showcase selection")
    selection = select_hybrid_showcase_case(
        aggressive_records=_jsonl(inputs["aggressive_episodes"]),
        defensive_records=_jsonl(inputs["defensive_episodes"]),
        explorer_records=_jsonl(inputs["explorer_episodes"]),
        defensive_telemetry=_jsonl(inputs["defensive_telemetry"]),
        explorer_telemetry=_jsonl(inputs["explorer_telemetry"]),
    )
    cases = load_duel_cases(
        cases_path,
        expected_split="validation",
        pairs_per_opponent=50,
    )
    selected = [
        case
        for case in cases
        if f"{case.opponent}:{case.pair_index}:{case.learner_side}"
        == selection["selected_case_id"]
    ]
    if len(selected) != 1:
        raise ValueError("Selected hybrid showcase case is not uniquely frozen")
    manifest = {
        "schema_version": 1,
        "source": _relative(cases_path, root),
        "source_sha256": sha256_file(cases_path),
        "cases": [selected[0].to_dict()],
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_bytes(canonical_json(manifest))
    evidence = {
        "schema_version": 1,
        "stage": "hybrid_product_showcase_selection",
        **selection,
        "source_sha256": {
            name: sha256_file(path) for name, path in inputs.items()
        },
        "case_manifest": _relative(output_manifest, root),
        "case_manifest_sha256": sha256_file(output_manifest),
    }
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_bytes(canonical_json(evidence))
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0
