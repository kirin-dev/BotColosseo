from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.cli.render_extraction_v3 import _representative_claims
from botcolosseo.data.demonstrations import sha256_file

POLICIES = ("strong", "aggressive", "defensive", "explorer")
EVIDENCE_TIERS = {
    "research_selection",
    "product_showcase",
    "directional_showcase",
    "validation_demonstration",
    "representative_case_demonstration",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the validation-only Extraction product Showcase"
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--board-manifest", type=Path, required=True)
    parser.add_argument(
        "--method",
        type=Path,
        default=Path("docs/assets/extraction/method.svg"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _load_bound_json(root: Path, relative: object, expected_sha256: object) -> dict:
    if not isinstance(relative, str) or not isinstance(expected_sha256, str):
        raise ValueError("Showcase artifact binding is missing")
    path = root / relative
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"Showcase artifact drifted: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_tier(
    policy: str,
    checkpoint_sha256: object,
    manifest: dict[str, object],
) -> str:
    if (
        manifest.get("gate_schema_version") == 2
        and manifest.get("eligible") is True
        and manifest.get("selected_checkpoint_sha256") == checkpoint_sha256
    ):
        tier = "research_selection"
    elif (
        policy == "strong"
        and manifest.get("admission_kind") == "strong_product_showcase"
        and manifest.get("product_showcase_eligible") is True
        and manifest.get("research_gate_passed") is False
        and manifest.get("official_test_eligible") is False
        and manifest.get("showcase_checkpoint_sha256") == checkpoint_sha256
    ):
        tier = "product_showcase"
    elif (
        manifest.get("admission_kind") == "directional_showcase"
        and manifest.get("showcase_eligible") is True
        and manifest.get("research_gate_passed") is False
        and manifest.get("showcase_checkpoint_sha256") == checkpoint_sha256
    ):
        tier = "directional_showcase"
    elif (
        manifest.get("evidence_tier") == "validation_demonstration"
        and manifest.get("product_demo_eligible") is True
        and manifest.get("research_gate_passed") is False
        and manifest.get("official_test_eligible") is False
        and manifest.get("showcase_checkpoint_sha256") == checkpoint_sha256
    ):
        tier = "validation_demonstration"
    elif (
        manifest.get("evidence_tier") == "representative_case_demonstration"
        and manifest.get("product_demo_eligible") is True
        and manifest.get("research_gate_passed") is False
        and manifest.get("official_test_eligible") is False
        and manifest.get("claim_scope")
        == "representative_validation_cases_only"
        and manifest.get("aggregate_style_gate_passed") is False
        and isinstance(manifest.get("representative_case_count"), int)
        and manifest["representative_case_count"] > 0
        and manifest.get("showcase_checkpoint_sha256") == checkpoint_sha256
    ):
        tier = "representative_case_demonstration"
    else:
        raise ValueError(f"{policy} Showcase artifact manifest is invalid")
    if (
        manifest.get("policy") != policy
        or manifest.get("test_cases_accessed") is not False
    ):
        raise ValueError(f"{policy} Showcase artifact identity does not match")
    return tier


def audit_extraction_showcase(
    *,
    root: Path,
    selection_path: Path,
    board_manifest_path: Path,
    method_path: Path,
) -> dict[str, object]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    board = json.loads(board_manifest_path.read_text(encoding="utf-8"))
    if (
        selection.get("schema_version") != 2
        or selection.get("selection_split") != "validation"
        or selection.get("test_cases_accessed") is not False
        or set(selection.get("selections", {})) != set(POLICIES)
    ):
        raise ValueError("Showcase selection identity does not match")
    if (
        board.get("schema_version") != 1
        or board.get("source_split") != "validation"
        or board.get("test_cases_accessed") is not False
        or set(board.get("artifacts", {})) != set(POLICIES)
    ):
        raise ValueError("Showcase board identity does not match")
    if (
        board.get("selection") != str(selection_path.relative_to(root))
        or board.get("selection_sha256") != sha256_file(selection_path)
    ):
        raise ValueError("Showcase board selection binding drifted")
    board_path = root / str(board.get("board"))
    if (
        not board_path.is_file()
        or sha256_file(board_path) != board.get("board_sha256")
    ):
        raise ValueError("Showcase board drifted")
    if not method_path.is_file():
        raise ValueError("Showcase method diagram is missing")

    audited: dict[str, object] = {}
    tiers: dict[str, str] = {}
    for policy in POLICIES:
        selected = selection["selections"][policy]
        artifact = board["artifacts"][policy]
        tier = selected.get("evidence_tier")
        if tier not in EVIDENCE_TIERS:
            raise ValueError(f"{policy} Showcase evidence tier is invalid")
        if artifact.get("evidence_tier") != tier:
            raise ValueError(f"{policy} Showcase board tier drifted")
        manifest = _load_bound_json(
            root,
            selected.get("artifact_manifest"),
            selected.get("artifact_manifest_sha256"),
        )
        if (
            _manifest_tier(
                policy,
                selected.get("checkpoint_sha256"),
                manifest,
            )
            != tier
        ):
            raise ValueError(f"{policy} Showcase evidence tier drifted")
        evidence = _load_bound_json(
            root,
            artifact.get("evidence"),
            artifact.get("evidence_sha256"),
        )
        video = root / str(artifact.get("video"))
        if (
            not video.is_file()
            or sha256_file(video) != artifact.get("video_sha256")
            or evidence.get("media") != artifact.get("video")
            or evidence.get("media_sha256") != artifact.get("video_sha256")
        ):
            raise ValueError(f"{policy} Showcase media drifted")
        if (
            evidence.get("policy") != policy
            or evidence.get("case_index") != selected.get("case_index")
            or evidence.get("checkpoint_sha256")
            != selected.get("checkpoint_sha256")
            or evidence.get("test_cases_accessed") is not False
        ):
            raise ValueError(f"{policy} Showcase replay identity does not match")
        capture_mode = evidence.get("capture_mode", "live_render")
        if capture_mode == "verified_existing_live_capture":
            source_evidence_sha256 = evidence.get("source_evidence_sha256")
            if (
                evidence.get("source_media_sha256")
                != evidence.get("media_sha256")
                or not evidence.get("source_media")
                or not evidence.get("source_evidence")
                or not isinstance(source_evidence_sha256, str)
                or len(source_evidence_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in source_evidence_sha256
                )
            ):
                raise ValueError(f"{policy} verified capture provenance is invalid")
        elif capture_mode != "live_render":
            raise ValueError(f"{policy} Showcase capture mode is invalid")
        claims = evidence.get("showcase_claims")
        if not isinstance(claims, dict):
            raise ValueError(f"{policy} Showcase claims are missing")
        representative, failures = _representative_claims(policy, claims)
        if not representative:
            raise ValueError(
                f"{policy} Showcase story is incomplete: {failures}"
            )
        attempt_count = evidence.get("render_attempt_count")
        attempts = evidence.get("render_attempts")
        if (
            not isinstance(attempt_count, int)
            or not 1 <= attempt_count <= 5
            or not isinstance(attempts, list)
            or len(attempts) != attempt_count
            or attempts[-1].get("accepted") is not True
            or any(item.get("accepted") is True for item in attempts[:-1])
        ):
            raise ValueError(f"{policy} Showcase attempt ledger is invalid")
        frame_count = evidence.get("frame_count")
        fps = evidence.get("fps")
        if (
            not isinstance(frame_count, int)
            or not isinstance(fps, int)
            or fps <= 0
        ):
            raise ValueError(f"{policy} Showcase duration metadata is invalid")
        duration = frame_count / fps
        if not 20 <= duration <= 60:
            raise ValueError(
                f"{policy} Showcase duration is outside 20-60 seconds"
            )
        complete_case_study = (
            tier == "representative_case_demonstration"
            and selected.get("case_selection_mode")
            == "complete_validation_case_study"
        )
        if (
            policy != "strong"
            and selected.get("paired_style_difference", 0) <= 0
            and not complete_case_study
        ):
            raise ValueError(f"{policy} Showcase evidence is not validation-safe")
        tiers[policy] = str(tier)
        validation_checks = manifest.get("validation_checks", [])
        if not validation_checks:
            validation_checks = manifest.get("checks", [])
        heldout_checks = manifest.get(
            "heldout_checks",
            manifest.get("original_heldout_checks", []),
        )
        style_difference = next(
            (
                check.get("value")
                for check in validation_checks
                if check.get("name") == "style_paired_difference"
            ),
            None,
        )
        audited[policy] = {
            "case_index": evidence["case_index"],
            "case_selection_mode": selected.get(
                "case_selection_mode", "paired_directional_case"
            ),
            "duration_seconds": duration,
            "evidence_tier": tier,
            "research_gate_passed": tier == "research_selection",
            "artifact_official_test_eligible": tier == "research_selection",
            "disclosed_failed_checks": manifest.get(
                "research_failed_checks",
                [],
            ),
            "validation_checks": validation_checks,
            "heldout_checks": heldout_checks,
            "direction_counts": manifest.get("direction_counts"),
            "validation_style_paired_difference": style_difference,
            "render_attempt_count": attempt_count,
            "capture_mode": capture_mode,
            "story_checks_passed": True,
        }
    return {
        "schema_version": 1,
        "passed": True,
        "selection": str(selection_path.relative_to(root)),
        "selection_sha256": sha256_file(selection_path),
        "board_manifest": str(board_manifest_path.relative_to(root)),
        "board_manifest_sha256": sha256_file(board_manifest_path),
        "method": str(method_path.relative_to(root)),
        "method_sha256": sha256_file(method_path),
        "policies": audited,
        "evidence_tiers": tiers,
        "source_split": "validation",
        "official_test_eligible": False,
        "test_cases_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    output = _resolve(root, args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Showcase audit: {output}")
    result = audit_extraction_showcase(
        root=root,
        selection_path=_resolve(root, args.selection),
        board_manifest_path=_resolve(root, args.board_manifest),
        method_path=_resolve(root, args.method),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
