from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.evaluation.extraction import ExtractionEpisodeMetrics
from botcolosseo.evaluation.extraction_gates import style_validation_gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank Extraction candidates using validation evidence only"
    )
    parser.add_argument(
        "--policy",
        choices=("strong", "aggressive", "defensive", "explorer"),
        required=True,
    )
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--strong-validation-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _load(path: Path, policy: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("policy") != policy
        or payload.get("metric_schema_version") != 2
        or payload.get("split") != "validation"
        or payload.get("complete") is not True
        or payload.get("test_cases_accessed") is not False
        or payload.get("actor_privilege_violations") != 0
        or payload.get("fair_actor_observation_only") is not True
    ):
        raise ValueError("Candidate ranking report identity does not match")
    if policy == "defensive" and payload.get("disengagement_metric_version") != 3:
        raise ValueError("Defensive report uses the legacy disengagement metric")
    if len(payload["metrics"]["episodes"]) != 240:
        raise ValueError("Candidate ranking requires 240 validation episodes")
    return payload


def _episodes(payload: dict[str, object]) -> tuple[ExtractionEpisodeMetrics, ...]:
    return tuple(
        ExtractionEpisodeMetrics(**item)
        for item in payload["metrics"]["episodes"]
    )


def _strong_score(report: dict[str, object]) -> tuple[float, ...]:
    metrics = report["metrics"]
    worst = min(
        float(item["win_rate"]) for item in metrics["by_opponent"].values()
    )
    values = (
        float(metrics["win_rate"]),
        worst,
        float(metrics["extraction_rate"]),
        float(metrics["mean_extracted_value_advantage"]),
        -float(metrics["protocol_inconsistencies"]),
    )
    passed = sum(
        (
            values[0] >= 0.70,
            values[1] >= 0.55,
            values[2] >= 0.75,
            values[3] > 0,
            values[4] == 0,
        )
    )
    return (float(passed), *values)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    output = _resolve(root, args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite candidate ranking: {output}")
    reports = [(_resolve(root, path), None) for path in args.report]
    strong_report = None
    if args.policy == "strong":
        if args.strong_validation_report is not None:
            raise ValueError("Strong ranking does not accept paired Strong evidence")
    else:
        if args.strong_validation_report is None:
            raise ValueError("Style ranking requires paired Strong evidence")
        strong_path = _resolve(root, args.strong_validation_report)
        strong_report = _load(strong_path, "strong")
    candidates: list[dict[str, object]] = []
    protocol_sha256 = None
    for report_path, _ in reports:
        report = _load(report_path, args.policy)
        if protocol_sha256 is None:
            protocol_sha256 = report["protocol_sha256"]
        if report["protocol_sha256"] != protocol_sha256:
            raise ValueError("Candidate ranking protocols do not match")
        checkpoint = root / report["checkpoint"]
        if sha256_file(checkpoint) != report["checkpoint_sha256"]:
            raise ValueError("Candidate ranking checkpoint hash drifted")
        if args.policy == "strong":
            score = _strong_score(report)
            eligible = score[0] == 5
        else:
            if strong_report["protocol_sha256"] != protocol_sha256:
                raise ValueError("Paired Strong ranking protocol does not match")
            gate = style_validation_gate(
                style=args.policy,
                strong=_episodes(strong_report),
                styled=_episodes(report),
            )
            score = (
                float(sum(check.passed for check in gate.checks)),
                next(
                    check.value
                    for check in gate.checks
                    if check.name == "style_ci_lower"
                ),
                next(
                    check.value
                    for check in gate.checks
                    if check.name == "paired_task_retention"
                ),
            )
            eligible = gate.passed
        candidates.append(
            {
                "report": str(report_path.relative_to(root)),
                "report_sha256": sha256_file(report_path),
                "checkpoint": report["checkpoint"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "eligible": eligible,
                "score": list(score),
            }
        )
    if args.policy == "strong":
        frontier = candidates
        selected = max(candidates, key=lambda item: tuple(item["score"]))
    else:
        eligible = [item for item in candidates if item["eligible"]]
        pool = eligible or candidates

        def objectives(item: dict[str, object]) -> tuple[float, float]:
            score = item["score"]
            return float(score[1]), float(score[2])

        frontier = [
            candidate
            for candidate in pool
            if not any(
                other is not candidate
                and objectives(other)[0] >= objectives(candidate)[0]
                and objectives(other)[1] >= objectives(candidate)[1]
                and objectives(other) != objectives(candidate)
                for other in pool
            )
        ]
        selected = max(frontier, key=lambda item: objectives(item))
    payload = {
        "schema_version": 1,
        "policy": args.policy,
        "selection_split": "validation",
        "protocol_sha256": protocol_sha256,
        "candidates": candidates,
        "pareto_frontier": [
            {
                "checkpoint": item["checkpoint"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "eligible": item["eligible"],
                "score": item["score"],
            }
            for item in frontier
        ],
        "selected": selected,
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
