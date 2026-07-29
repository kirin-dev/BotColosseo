from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.evaluation.extraction import ExtractionEpisodeMetrics
from botcolosseo.evaluation.extraction_gates import style_validation_gate

STYLES = ("aggressive", "defensive", "explorer")
VARIANTS = ("full", "reward-plus-kl", "reward-only")
COEFFICIENTS = {
    "full": (0.08, 0.01),
    "reward-plus-kl": (0.08, 0.00),
    "reward-only": (0.00, 0.00),
}
FULL_OUTPUTS = {
    "aggressive": Path("runs/extraction/styles/aggressive"),
    "defensive": Path(
        "runs/extraction/styles/defensive-calibration-v2"
    ),
    "explorer": Path("runs/extraction/styles/explorer"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize the frozen 200k Extraction style ablation matrix"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/extraction/style-ablation.json"),
    )
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _episodes(report: dict[str, object]) -> tuple[ExtractionEpisodeMetrics, ...]:
    return tuple(
        ExtractionEpisodeMetrics(**item)
        for item in report["metrics"]["episodes"]
    )


def _strong_validation_path(root: Path) -> Path:
    selection = json.loads(
        (root / "runs/extraction/strong-ppo/selection.json").read_text(
            encoding="utf-8"
        )
    )
    matches = [
        root / path
        for path in selection["evidence"]
        if path.endswith("-validation.json")
    ]
    if len(matches) != 1:
        raise ValueError("Strong selection has no unique validation report")
    return matches[0]


def _cell_paths(root: Path, variant: str, style: str) -> tuple[Path, Path]:
    if variant == "full":
        output = root / FULL_OUTPUTS[style]
        evaluation = output / "evaluation-v2"
    else:
        output = root / "runs/extraction/ablations" / variant / style
        evaluation = output / "evaluation"
    checkpoint = output / "candidate-0200000.pt"
    report = evaluation / "candidate-0200000-validation.json"
    return checkpoint, report


def _load_report(
    path: Path,
    *,
    policy: str,
    checkpoint: Path,
    base_sha256: str,
    protocol_sha256: str,
    scenario_hash: str,
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("policy") != policy
        or payload.get("split") != "validation"
        or payload.get("metric_schema_version") != 2
        or payload.get("complete") is not True
        or payload.get("episodes_evaluated") != 240
        or payload.get("expected_episodes") != 240
        or payload.get("test_cases_accessed") is not False
        or payload.get("actor_privilege_violations") != 0
        or payload.get("fair_actor_observation_only") is not True
        or payload.get("base_checkpoint_sha256") != base_sha256
        or payload.get("protocol_sha256") != protocol_sha256
        or payload.get("scenario_hash") != scenario_hash
        or payload.get("checkpoint_sha256") != sha256_file(checkpoint)
        or Path(str(payload.get("checkpoint"))).name
        != "candidate-0200000.pt"
    ):
        raise ValueError(f"Ablation report identity does not match: {path}")
    return payload


def _config_payload(
    root: Path,
    checkpoint: Path,
    *,
    variant: str,
    style: str,
    base_sha256: str,
) -> tuple[dict[str, object], Path]:
    summary_path = checkpoint.parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config_path = root / str(summary["config"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    beta_kl, rho_residual = COEFFICIENTS[variant]
    expected_horizon = 400_000 if style == "defensive" else 600_000
    expected_overrides = (
        {
            "risk_disengagement": 0.30,
            "combat_with_value": -0.030,
        }
        if style == "defensive"
        else None
    )
    actual_overrides = config.get("style_reward_overrides", {}).get(style)
    if (
        summary.get("style") != style
        or summary.get("base_checkpoint_sha256") != base_sha256
        or summary.get("fair_actor_observation_only") is not True
        or summary.get("frozen_strong_actor") is not True
        or summary.get("frozen_strong_base") is not True
        or summary.get("learned_residual_adapter") is not True
        or summary.get("test_cases_accessed") is not False
        or float(config["beta_kl"]) != beta_kl
        or float(config["rho_residual"]) != rho_residual
        or int(config["environment_steps"]) != expected_horizon
        or float(config["style_reward_scale"][style]) != 1.0
        or actual_overrides != expected_overrides
    ):
        raise ValueError(
            f"Ablation training configuration does not match: {summary_path}"
        )
    return summary, config_path


def _check_map(gate) -> dict[str, dict[str, object]]:
    return {
        check.name: {
            "passed": check.passed,
            "value": check.value,
            "threshold": check.threshold,
        }
        for check in gate.checks
    }


def summarize(root: Path) -> dict[str, object]:
    strong_path = _strong_validation_path(root)
    strong_report = json.loads(strong_path.read_text(encoding="utf-8"))
    if (
        strong_report.get("policy") != "strong"
        or strong_report.get("split") != "validation"
        or strong_report.get("complete") is not True
        or strong_report.get("episodes_evaluated") != 240
        or strong_report.get("test_cases_accessed") is not False
        or strong_report.get("actor_privilege_violations") != 0
        or strong_report.get("fair_actor_observation_only") is not True
    ):
        raise ValueError("Strong paired validation identity does not match")
    strong = _episodes(strong_report)
    base_sha256 = str(strong_report["checkpoint_sha256"])
    protocol_sha256 = str(strong_report["protocol_sha256"])
    scenario_hash = str(strong_report["scenario_hash"])
    matrix: dict[str, dict[str, object]] = {}
    for variant in VARIANTS:
        cells: dict[str, object] = {}
        for style in STYLES:
            checkpoint, report_path = _cell_paths(root, variant, style)
            if not checkpoint.is_file() or not report_path.is_file():
                raise FileNotFoundError(
                    f"Ablation cell is incomplete: {variant}/{style}"
                )
            report = _load_report(
                report_path,
                policy=style,
                checkpoint=checkpoint,
                base_sha256=base_sha256,
                protocol_sha256=protocol_sha256,
                scenario_hash=scenario_hash,
            )
            summary, config_path = _config_payload(
                root,
                checkpoint,
                variant=variant,
                style=style,
                base_sha256=base_sha256,
            )
            gate = style_validation_gate(
                style=style,
                strong=strong,
                styled=_episodes(report),
            )
            checks = _check_map(gate)
            cells[style] = {
                "beta_kl": COEFFICIENTS[variant][0],
                "rho_residual": COEFFICIENTS[variant][1],
                "checkpoint": str(checkpoint.relative_to(root)),
                "checkpoint_sha256": sha256_file(checkpoint),
                "config": str(config_path.relative_to(root)),
                "config_sha256": sha256_file(config_path),
                "training_summary": str(
                    (checkpoint.parent / "summary.json").relative_to(root)
                ),
                "training_summary_sha256": sha256_file(
                    checkpoint.parent / "summary.json"
                ),
                "training_environment_steps_observed": summary[
                    "environment_steps"
                ],
                "validation_report": str(report_path.relative_to(root)),
                "validation_report_sha256": sha256_file(report_path),
                "gate_passed": gate.passed,
                "paired_style_shift": checks["style_paired_difference"][
                    "value"
                ],
                "style_ci_lower": checks["style_ci_lower"]["value"],
                "style_ci_upper": checks["style_ci_upper"]["value"],
                "paired_task_retention": checks["paired_task_retention"][
                    "value"
                ],
                "extraction_rate_delta": checks["extraction_rate_delta"][
                    "value"
                ],
                "mean_value_ratio": checks["mean_value_ratio"]["value"],
                "worst_opponent_retention": checks[
                    "worst_opponent_retention"
                ]["value"],
                "checks": checks,
            }
        matrix[variant] = cells
    return {
        "schema_version": 1,
        "comparison_environment_steps": 200_000,
        "selection_split": "validation",
        "styles": list(STYLES),
        "variants": list(VARIANTS),
        "strong_checkpoint_sha256": base_sha256,
        "strong_validation_report": str(strong_path.relative_to(root)),
        "strong_validation_report_sha256": sha256_file(strong_path),
        "protocol_sha256": protocol_sha256,
        "scenario_hash": scenario_hash,
        "matrix": matrix,
        "actor_privilege_violations": 0,
        "test_cases_accessed": False,
        "official_test_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    output = _resolve(root, args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite ablation summary: {output}")
    payload = summarize(root)
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
