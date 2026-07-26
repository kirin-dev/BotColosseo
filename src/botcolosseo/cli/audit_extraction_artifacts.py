from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.agents.extraction_teachers import ExtractionStyle
from botcolosseo.data.demonstrations import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Extraction v2 data, checkpoints, and validation reports"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/generated/extraction-v2"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/extraction-v2"),
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("reports/extraction-v2/validation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/extraction-v2/training-artifact-audit.json"),
    )
    parser.add_argument("--expected-train-transitions", type=int, default=60000)
    parser.add_argument("--expected-validation-transitions", type=int, default=12000)
    return parser


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def audit_extraction_artifacts(
    *,
    root: Path,
    data_root: Path,
    run_root: Path,
    report_root: Path,
    expected_train_transitions: int,
    expected_validation_transitions: int,
) -> dict[str, object]:
    scenario_hash = _load_json(
        root / "assets/scenarios/crystal_run_extraction/manifest.json"
    )["wad_sha256"]
    if not isinstance(scenario_hash, str):
        raise ValueError("Extraction scenario hash is missing")
    rows: dict[str, object] = {}
    strong_hash = ""
    for style in ExtractionStyle:
        data: dict[str, object] = {}
        for split, expected in (
            ("train", expected_train_transitions),
            ("validation", expected_validation_transitions),
        ):
            manifest_path = (
                data_root / style.value / split / f"{split}-manifest.json"
            )
            manifest = _load_json(manifest_path)
            if (
                manifest.get("style") != style.value
                or manifest.get("split") != split
                or manifest.get("transitions") != expected
                or manifest.get("scenario_hash") != scenario_hash
                or manifest.get("test_cases_accessed") is not False
            ):
                raise ValueError(
                    f"Extraction {style.value}/{split} manifest invariants failed"
                )
            shard_total = 0
            for item in manifest["shards"]:  # type: ignore[union-attr]
                shard = manifest_path.parent / item["file"]
                if sha256_file(shard) != item["sha256"]:
                    raise ValueError(f"Extraction shard hash mismatch: {shard}")
                shard_total += int(item["transitions"])
            if shard_total != expected:
                raise ValueError("Extraction shard transition total does not match")
            data[split] = {
                "manifest_sha256": sha256_file(manifest_path),
                "transitions": expected,
            }

        summary_path = run_root / style.value / "summary.json"
        summary = _load_json(summary_path)
        checkpoint = root / str(summary["best_checkpoint"])
        if (
            summary.get("style") != style.value
            or summary.get("scenario_hash") != scenario_hash
            or summary.get("test_cases_accessed") is not False
            or summary.get("fair_observation_only") is not True
            or sha256_file(checkpoint) != summary.get("checkpoint_sha256")
        ):
            raise ValueError(f"Extraction {style.value} checkpoint audit failed")
        if style is ExtractionStyle.STRONG:
            strong_hash = str(summary["checkpoint_sha256"])
            if summary.get("residual_style_branch") is not False:
                raise ValueError("Extraction Strong Base cannot be a style branch")
        elif (
            summary.get("residual_style_branch") is not True
            or summary.get("base_checkpoint_sha256") != strong_hash
        ):
            raise ValueError(
                f"Extraction {style.value} is not bound to Strong Base"
            )

        report_path = report_root / f"{style.value}.json"
        report = _load_json(report_path)
        metrics = report.get("metrics")
        if (
            report.get("style") != style.value
            or report.get("scenario_hash") != scenario_hash
            or report.get("checkpoint_sha256") != summary["checkpoint_sha256"]
            or report.get("split") != "validation"
            or report.get("test_cases_accessed") is not False
            or not isinstance(metrics, dict)
            or len(metrics.get("episodes", [])) != 4
        ):
            raise ValueError(f"Extraction {style.value} validation audit failed")
        rows[style.value] = {
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "data": data,
            "summary_sha256": sha256_file(summary_path),
            "validation_report_sha256": sha256_file(report_path),
        }
    return {
        "artifact_gate_passed": True,
        "policies": rows,
        "scenario_hash": scenario_hash,
        "schema_version": 1,
        "stage": "extraction-v2-training-artifacts",
        "test_cases_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    output = _resolve(args.output, root)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite artifact audit: {output}")
    result = audit_extraction_artifacts(
        root=root,
        data_root=_resolve(args.data_root, root),
        run_root=_resolve(args.run_root, root),
        report_root=_resolve(args.report_root, root),
        expected_train_transitions=args.expected_train_transitions,
        expected_validation_transitions=args.expected_validation_transitions,
    )
    _atomic_json(result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
