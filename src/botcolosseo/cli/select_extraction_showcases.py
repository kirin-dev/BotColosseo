from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.evaluation.extraction import ExtractionEpisodeMetrics
from botcolosseo.evaluation.extraction_protocol import (
    load_extraction_evaluation_protocol,
)

POLICIES = ("strong", "aggressive", "defensive", "explorer")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select representative validation-only Extraction replays"
    )
    for policy in POLICIES:
        parser.add_argument(f"--{policy}-report", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/extraction/evaluation.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _score(policy: str, episode: ExtractionEpisodeMetrics) -> tuple[int, ...]:
    duration_ok = int(episode.decisions >= 240)
    if policy == "strong":
        return (
            int(episode.extracted),
            duration_ok,
            int(episode.won),
            episode.extracted_value,
            episode.valid_hits,
        )
    if policy == "aggressive":
        chain = (
            episode.valid_hits >= 5
            and episode.kills >= 1
            and episode.cache_looted >= 1
            and episode.extracted
        )
        return (
            int(chain),
            int(episode.extracted),
            duration_ok,
            episode.kills,
            episode.cache_looted,
            episode.valid_hits,
            episode.extracted_value,
        )
    if policy == "defensive":
        return (
            int(episode.extracted and not episode.died),
            duration_ok,
            -episode.attack_decisions,
            episode.extracted_value,
            -episode.valid_hits,
        )
    if policy == "explorer":
        return (
            int(episode.extracted),
            duration_ok,
            episode.unique_route_cells,
            episode.loot_pickups,
            episode.extracted_value,
        )
    raise ValueError("Unknown Extraction showcase policy")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    protocol_path = _resolve(root, args.protocol)
    output = _resolve(root, args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite showcase selection: {output}")
    protocol = load_extraction_evaluation_protocol(protocol_path)
    cases = protocol.cases("validation")
    case_indices = {
        (case.seed, case.learner_side, case.opponent_style): index
        for index, case in enumerate(cases)
    }
    selections: dict[str, object] = {}
    for policy in POLICIES:
        report_path = _resolve(root, getattr(args, f"{policy}_report"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("policy") != policy
            or report.get("split") != "validation"
            or report.get("complete") is not True
            or report.get("protocol_sha256") != protocol.sha256
            or report.get("test_cases_accessed") is not False
        ):
            raise ValueError(f"{policy} showcase report identity does not match")
        episodes = tuple(
            ExtractionEpisodeMetrics(**item)
            for item in report["metrics"]["episodes"]
        )
        selected = max(episodes, key=lambda item: _score(policy, item))
        identity = (
            selected.seed,
            selected.learner_side,
            selected.opponent_style,
        )
        selections[policy] = {
            "case_index": case_indices[identity],
            "episode": vars(selected),
            "score": list(_score(policy, selected)),
            "validation_report": str(report_path.relative_to(root)),
            "validation_report_sha256": sha256_file(report_path),
            "checkpoint_sha256": report["checkpoint_sha256"],
        }
    payload = {
        "schema_version": 1,
        "protocol": str(protocol_path.relative_to(root)),
        "protocol_sha256": protocol.sha256,
        "selection_split": "validation",
        "selections": selections,
        "test_cases_accessed": False,
    }
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
