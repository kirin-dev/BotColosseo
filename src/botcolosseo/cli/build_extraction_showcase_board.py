from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from botcolosseo.data.demonstrations import sha256_file

POLICIES = ("strong", "aggressive", "defensive", "explorer")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the four-policy Extraction GitHub showcase board"
    )
    for policy in POLICIES:
        parser.add_argument(f"--{policy}-video", type=Path, required=True)
        parser.add_argument(f"--{policy}-evidence", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _representative_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            raise ValueError(f"Showcase video has no frames: {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, round(count * 0.65)))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Cannot decode showcase video: {path}")
        return cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
    finally:
        capture.release()


def _caption(policy: str, evidence: dict[str, object]) -> str:
    claims = evidence["showcase_claims"]
    if policy == "strong":
        return f"Banked {claims['extracted_value']} value"
    if policy == "aggressive":
        return (
            f"Kill-cache-extract chain x{claims['aggressive_chains']}"
        )
    if policy == "defensive":
        return (
            f"Disengages x{claims['successful_disengagements']} | "
            f"banks {claims['extracted_value']}"
        )
    return (
        f"Upgrade-extract x{claims['upgrade_to_extraction_conversions']} | "
        f"banks {claims['extracted_value']}"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    output = _resolve(root, args.output)
    manifest_path = _resolve(root, args.manifest)
    selection_path = _resolve(root, args.selection)
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite showcase board artifacts")
    canvas = np.full((790, 1080, 3), (15, 20, 28), dtype=np.uint8)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection.get("schema_version") != 2
        or selection.get("selection_split") != "validation"
        or selection.get("test_cases_accessed") is not False
    ):
        raise ValueError("Showcase selection identity does not match")
    cv2.putText(
        canvas,
        "BOTCOLOSSEO: SEARCH-FIGHT-EXTRACT",
        (30, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "One Strong Base. Three learned behavioral styles.",
        (30, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (170, 190, 210),
        1,
        cv2.LINE_AA,
    )
    artifacts: dict[str, object] = {}
    for index, policy in enumerate(POLICIES):
        video = _resolve(root, getattr(args, f"{policy}_video"))
        evidence_path = _resolve(root, getattr(args, f"{policy}_evidence"))
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        selected = selection["selections"][policy]
        evidence_tier = selected["evidence_tier"]
        if (
            evidence.get("policy") != policy
            or evidence.get("test_cases_accessed") is not False
        ):
            raise ValueError(f"{policy} showcase evidence identity does not match")
        frame = _representative_frame(video)
        row, column = divmod(index, 2)
        x, y = 20 + column * 530, 90 + row * 345
        frame_x, frame_y = x + 15, y + 38
        cv2.rectangle(
            canvas,
            (x, y),
            (x + 510, y + 330),
            (100, 120, 145),
            1,
        )
        canvas[frame_y : frame_y + 270, frame_x : frame_x + 480] = frame
        cv2.putText(
            canvas,
            policy.upper(),
            (x + 12, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (80, 220, 255),
            2,
            cv2.LINE_AA,
        )
        tier_label = {
            "research_selection": "RESEARCH SELECTED",
            "product_showcase": "PRODUCT SHOWCASE",
            "directional_showcase": "DIRECTIONAL SHOWCASE",
            "validation_demonstration": "VALIDATION DEMO",
            "representative_case_demonstration": "CASE STUDY",
        }.get(evidence_tier)
        if tier_label is None:
            raise ValueError(f"{policy} Showcase evidence tier is invalid")
        cv2.putText(
            canvas,
            tier_label,
            (x + 285, y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            _caption(policy, evidence),
            (x + 12, y + 323),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        artifacts[policy] = {
            "video": str(video.relative_to(root)),
            "video_sha256": sha256_file(video),
            "evidence": str(evidence_path.relative_to(root)),
            "evidence_sha256": sha256_file(evidence_path),
            "evidence_tier": evidence_tier,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError("Failed to write Extraction showcase board")
    manifest = {
        "schema_version": 1,
        "board": str(output.relative_to(root)),
        "board_sha256": sha256_file(output),
        "selection": str(selection_path.relative_to(root)),
        "selection_sha256": sha256_file(selection_path),
        "artifacts": artifacts,
        "source_split": "validation",
        "test_cases_accessed": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
