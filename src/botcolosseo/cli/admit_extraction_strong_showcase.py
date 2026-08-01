from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from botcolosseo.cli.select_extraction_candidate import (
    _atomic_json,
    _episodes,
    _load_report,
)
from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.evaluation.extraction_gates import strong_validation_gate

PRODUCT_MIN_HELDOUT_EXTRACTION = 0.65


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Admit a validation-capable Strong checkpoint for product Showcase use"
        )
    )
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _verify_checkpoint_report(
    checkpoint: Path,
    *reports: dict[str, object],
) -> None:
    digest = sha256_file(checkpoint)
    if any(report.get("checkpoint_sha256") != digest for report in reports):
        raise ValueError("Strong Showcase report checkpoint hash does not match")
    if len({str(report.get("protocol_sha256")) for report in reports}) != 1:
        raise ValueError("Strong Showcase report protocols do not match")
    if len({str(report.get("scenario_hash")) for report in reports}) != 1:
        raise ValueError("Strong Showcase report scenarios do not match")


def build_strong_showcase_admission(
    *,
    root: Path,
    ranking_path: Path,
    evaluation_root: Path,
) -> tuple[dict[str, object], Path]:
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    if (
        ranking.get("policy") != "strong"
        or ranking.get("selection_split") != "validation"
        or ranking.get("test_cases_accessed") is not False
    ):
        raise ValueError("Strong Showcase ranking identity does not match")
    frontier = {
        item.get("checkpoint")
        for item in ranking.get("pareto_frontier", [])
        if item.get("eligible") is True
    }
    considered: list[dict[str, object]] = []
    eligible: list[tuple[float, tuple[object, ...], dict[str, object], Path]] = []
    for candidate in ranking.get("candidates", []):
        checkpoint_name = candidate.get("checkpoint")
        validation_name = candidate.get("report")
        if (
            candidate.get("eligible") is not True
            or checkpoint_name not in frontier
            or not isinstance(checkpoint_name, str)
            or not isinstance(validation_name, str)
        ):
            continue
        checkpoint = _resolve(root, Path(checkpoint_name))
        validation_path = _resolve(root, Path(validation_name))
        tag = checkpoint.stem
        heldout_path = evaluation_root / f"{tag}-heldout.json"
        solo_path = evaluation_root / f"{tag}-solo.json"
        validation = _load_report(
            validation_path, policy="strong", split="validation"
        )
        heldout = _load_report(heldout_path, policy="strong", split="heldout")
        solo = _load_report(solo_path, policy="strong", split="solo")
        _verify_checkpoint_report(checkpoint, validation, heldout, solo)
        gate = strong_validation_gate(
            _episodes(validation),
            _episodes(heldout),
            _episodes(solo),
        )
        failed = [check.name for check in gate.checks if not check.passed]
        heldout_check = next(
            check for check in gate.checks if check.name == "heldout_extraction"
        )
        item = {
            "checkpoint": checkpoint_name,
            "checkpoint_sha256": sha256_file(checkpoint),
            "research_failed_checks": failed,
            "heldout_extraction": heldout_check.value,
            "validation_score": candidate.get("score"),
        }
        considered.append(item)
        score = candidate.get("score")
        if (
            failed == ["heldout_extraction"]
            and heldout_check.value >= PRODUCT_MIN_HELDOUT_EXTRACTION
            and isinstance(score, list)
        ):
            evidence = {
                "validation": validation_path,
                "heldout": heldout_path,
                "solo": solo_path,
            }
            item["evidence"] = {
                split: _relative(root, path) for split, path in evidence.items()
            }
            eligible.append((heldout_check.value, tuple(score), item, checkpoint))
    if not eligible:
        raise ValueError("No Strong candidate passes the product Showcase rule")
    _, _, selected, checkpoint = max(eligible, key=lambda item: (item[0], item[1]))
    evidence_paths = {
        "ranking": ranking_path,
        **{
            split: _resolve(root, Path(path))
            for split, path in selected["evidence"].items()
        },
    }
    validation = _load_report(
        evidence_paths["validation"], policy="strong", split="validation"
    )
    heldout = _load_report(
        evidence_paths["heldout"], policy="strong", split="heldout"
    )
    solo = _load_report(evidence_paths["solo"], policy="strong", split="solo")
    gate = strong_validation_gate(
        _episodes(validation), _episodes(heldout), _episodes(solo)
    )
    result = {
        "schema_version": 1,
        "admission_schema_version": 1,
        "admission_kind": "strong_product_showcase",
        "admission_rule_timing": "post_heldout_product_review",
        "policy": "strong",
        "product_showcase_eligible": True,
        "research_gate_passed": False,
        "official_test_eligible": False,
        "research_failed_checks": [
            check.name for check in gate.checks if not check.passed
        ],
        "product_min_heldout_extraction": PRODUCT_MIN_HELDOUT_EXTRACTION,
        "research_checks": [asdict(check) for check in gate.checks],
        "candidate_choice_basis": (
            "highest_heldout_extraction_on_validation_pareto_frontier"
        ),
        "considered_candidates": considered,
        "candidate_checkpoint": _relative(root, checkpoint),
        "candidate_checkpoint_sha256": sha256_file(checkpoint),
        "evidence": [
            _relative(root, path) for path in evidence_paths.values()
        ],
        "evidence_sha256": {
            _relative(root, path): sha256_file(path)
            for path in evidence_paths.values()
        },
        "protocol_sha256": validation["protocol_sha256"],
        "scenario_hash": validation["scenario_hash"],
        "actor_privilege_violations": 0,
        "test_cases_accessed": False,
    }
    return result, checkpoint


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    ranking_path = _resolve(root, args.ranking)
    evaluation_root = _resolve(root, args.evaluation_root)
    output_checkpoint = _resolve(root, args.output_checkpoint)
    output_report = _resolve(root, args.output_report)
    if output_checkpoint.exists() or output_report.exists():
        raise FileExistsError("Refusing to overwrite a Strong Showcase admission")
    result, checkpoint = build_strong_showcase_admission(
        root=root,
        ranking_path=ranking_path,
        evaluation_root=evaluation_root,
    )
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_checkpoint.with_name(f".{output_checkpoint.name}.tmp")
    shutil.copyfile(checkpoint, temporary)
    temporary.replace(output_checkpoint)
    result["showcase_checkpoint"] = _relative(root, output_checkpoint)
    result["showcase_checkpoint_sha256"] = sha256_file(output_checkpoint)
    _atomic_json(result, output_report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
