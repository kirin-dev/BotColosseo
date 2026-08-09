from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from botcolosseo.cli.render_extraction_v3 import _representative_claims
from botcolosseo.cli.select_extraction_showcases import _representative, _score
from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.evaluation.extraction import ExtractionEpisodeMetrics
from botcolosseo.evaluation.extraction_gates import _style_score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adopt a verified existing validation Showcase capture"
    )
    parser.add_argument(
        "--policy",
        choices=("strong", "aggressive", "defensive", "explorer"),
        required=True,
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--failed-preselected-attempts", type=int, default=0)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing stale capture temporary: {temporary}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    selection_path = _resolve(root, args.selection)
    source_video = _resolve(root, args.source_video)
    source_evidence_path = _resolve(root, args.source_evidence)
    checkpoint = _resolve(root, args.checkpoint)
    base_checkpoint = (
        _resolve(root, args.base_checkpoint)
        if args.base_checkpoint is not None
        else None
    )
    output_video = _resolve(root, args.output_video)
    output_evidence = _resolve(root, args.output_evidence)
    if output_video.exists() or output_evidence.exists():
        raise FileExistsError("Refusing to overwrite final Showcase capture")
    if args.failed_preselected_attempts < 0:
        raise ValueError("Failed preselected attempts cannot be negative")
    if args.policy != "strong" and base_checkpoint is None:
        raise ValueError("Styled capture adoption requires a Strong Base")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    source = json.loads(source_evidence_path.read_text(encoding="utf-8"))
    selected = selection["selections"][args.policy]
    if (
        selection.get("schema_version") != 2
        or selection.get("selection_split") != "validation"
        or selection.get("test_cases_accessed") is not False
        or source.get("policy") != args.policy
        or source.get("test_cases_accessed") is not False
        or source.get("case", {}).get("split") != "validation"
        or source.get("checkpoint_sha256") != selected.get("checkpoint_sha256")
        or sha256_file(checkpoint) != selected.get("checkpoint_sha256")
        or (
            args.policy == "strong"
            and source.get("base_checkpoint_sha256") is not None
        )
        or (
            args.policy != "strong"
            and source.get("base_checkpoint_sha256")
            != sha256_file(base_checkpoint)
        )
        or source.get("media_sha256") != sha256_file(source_video)
    ):
        raise ValueError("Existing Showcase capture identity does not match")

    style_report = json.loads(
        (root / selected["validation_report"]).read_text(encoding="utf-8")
    )
    strong_selected = selection["selections"]["strong"]
    strong_report = json.loads(
        (root / strong_selected["validation_report"]).read_text(encoding="utf-8")
    )
    case_index = source.get("case_index")
    if (
        not isinstance(case_index, int)
        or not 0 <= case_index < len(style_report["metrics"]["episodes"])
        or len(style_report["metrics"]["episodes"])
        != len(strong_report["metrics"]["episodes"])
    ):
        raise ValueError("Existing Showcase capture case index does not match")
    episode = ExtractionEpisodeMetrics(
        **style_report["metrics"]["episodes"][case_index]
    )
    paired_strong = ExtractionEpisodeMetrics(
        **strong_report["metrics"]["episodes"][case_index]
    )
    identity = (episode.seed, episode.learner_side, episode.opponent_style)
    source_identity = (
        source["case"]["seed"],
        source["case"]["learner_side"],
        source["case"]["opponent_style"],
    )
    if (
        identity != source_identity
        or not _representative(args.policy, episode, paired_strong)
    ):
        raise ValueError("Existing Showcase capture is not validation representative")

    claims = dict(source["showcase_claims"])
    claims["decisions"] = source["decisions"]
    claims["died"] = source["died"]
    accepted, failures = _representative_claims(args.policy, claims)
    if not accepted:
        raise ValueError(f"Existing Showcase capture story is incomplete: {failures}")

    source_video_sha256 = sha256_file(source_video)
    source_evidence_sha256 = sha256_file(source_evidence_path)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    temporary_video = output_video.with_name(f".{output_video.name}.tmp")
    if temporary_video.exists():
        raise FileExistsError(f"Refusing stale capture temporary: {temporary_video}")
    shutil.copyfile(source_video, temporary_video)
    temporary_video.replace(output_video)
    evidence = {
        **source,
        "checkpoint": str(checkpoint.relative_to(root)),
        "base_checkpoint": (
            str(base_checkpoint.relative_to(root))
            if base_checkpoint is not None
            else None
        ),
        "media": str(output_video.relative_to(root)),
        "media_sha256": sha256_file(output_video),
        "showcase_claims": claims,
        "capture_mode": "verified_existing_live_capture",
        "source_media": str(source_video.relative_to(root)),
        "source_media_sha256": source_video_sha256,
        "source_evidence": str(source_evidence_path.relative_to(root)),
        "source_evidence_sha256": source_evidence_sha256,
        "render_attempt_count": 1,
        "render_attempts": [
            {
                "attempt": 1,
                "accepted": True,
                "failed_checks": [],
                "decisions": source["decisions"],
                "extracted": source["extracted"],
                "died": source["died"],
                "extracted_value": source["extracted_value"],
                "valid_hits": claims["valid_hits"],
                "kills": claims["kills"],
                "aggressive_chains": claims["aggressive_chains"],
                "successful_disengagements": claims[
                    "successful_disengagements"
                ],
                "backpack_upgrades": claims["backpack_upgrades"],
                "upgrade_to_extraction_conversions": claims[
                    "upgrade_to_extraction_conversions"
                ],
            }
        ],
    }
    _atomic_json(output_evidence, evidence)

    previous_case_index = selected["case_index"]
    if args.failed_preselected_attempts == 0:
        if case_index != previous_case_index:
            raise ValueError("Verified capture does not match the selected case")
    else:
        selected.update(
            {
                "case_index": case_index,
                "episode": vars(episode),
                "paired_strong_episode": vars(paired_strong),
                "paired_style_difference": _style_score(args.policy, episode)
                - _style_score(args.policy, paired_strong),
                "score": list(_score(args.policy, episode, paired_strong)),
                "case_selection_mode": "verified_existing_capture_fallback",
                "failed_preselected_case_indices": [previous_case_index],
                "failed_preselected_attempts": args.failed_preselected_attempts,
                "adopted_source_media_sha256": source_video_sha256,
                "adopted_source_evidence_sha256": source_evidence_sha256,
            }
        )
        selection["selection_revision"] = 1
        selection["previous_selection_sha256"] = sha256_file(selection_path)
        _atomic_json(selection_path, selection)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
