from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from botcolosseo.data.demonstrations import sha256_file

POLICIES = ("strong", "aggressive", "defensive", "explorer")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the complete Crystal Run: Extraction v3 release"
    )
    parser.add_argument(
        "--release-manifest",
        type=Path,
        default=Path("runs/extraction/release/manifest.json"),
    )
    parser.add_argument(
        "--official-receipt",
        type=Path,
        default=Path("runs/extraction/release/official-test/receipt.json"),
    )
    parser.add_argument(
        "--showcase-manifest",
        type=Path,
        default=Path("reports/extraction/showcase/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/extraction/release.json"),
    )
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _release_hash(payload: dict[str, object]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "release_sha256"}
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    release_path = _resolve(root, args.release_manifest)
    receipt_path = _resolve(root, args.official_receipt)
    showcase_path = _resolve(root, args.showcase_manifest)
    output = _resolve(root, args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite release audit: {output}")
    release = _load(release_path)
    receipt = _load(receipt_path)
    showcase = _load(showcase_path)
    current_scenario = _load(
        root / "assets/scenarios/crystal_run_extraction/manifest.json"
    )["wad_sha256"]
    if (
        release.get("release_sha256") != _release_hash(release)
        or release.get("scenario_hash") != current_scenario
        or release.get("test_cases_accessed") is not False
        or set(release.get("policies", {})) != set(POLICIES)
    ):
        raise ValueError("Release manifest identity does not match")
    for policy, spec in release["policies"].items():
        checkpoint = root / spec["checkpoint"]
        selection = root / spec["selection_report"]
        if (
            sha256_file(checkpoint) != spec["checkpoint_sha256"]
            or sha256_file(selection) != spec["selection_report_sha256"]
            or _load(selection).get("eligible") is not True
        ):
            raise ValueError(f"{policy} release selection drifted")
    if (
        receipt.get("complete") is not True
        or receipt.get("release_sha256") != release["release_sha256"]
        or receipt.get("protocol_sha256") != release["protocol_sha256"]
        or receipt.get("episodes_per_policy") != 400
        or receipt.get("total_episodes") != 1600
        or receipt.get("test_cases_accessed") is not True
        or set(receipt.get("policy_metrics", {})) != set(POLICIES)
    ):
        raise ValueError("Official-test receipt is incomplete or stale")
    if (
        showcase.get("source_split") != "validation"
        or showcase.get("test_cases_accessed") is not False
        or set(showcase.get("artifacts", {})) != set(POLICIES)
    ):
        raise ValueError("Showcase manifest identity does not match")
    board = root / showcase["board"]
    if sha256_file(board) != showcase["board_sha256"]:
        raise ValueError("Showcase board hash drifted")
    media_checks: dict[str, object] = {}
    for policy, artifact in showcase["artifacts"].items():
        video = root / artifact["video"]
        evidence_path = root / artifact["evidence"]
        if (
            sha256_file(video) != artifact["video_sha256"]
            or sha256_file(evidence_path) != artifact["evidence_sha256"]
        ):
            raise ValueError(f"{policy} showcase artifact hash drifted")
        evidence = _load(evidence_path)
        duration = evidence["frame_count"] / evidence["fps"]
        claims = evidence["showcase_claims"]
        passed = (
            evidence.get("policy") == policy
            and evidence.get("test_cases_accessed") is False
            and evidence.get("policy_kind")
            in {"strong-recurrent-ppo", "learned-bounded-residual"}
            and 20 <= duration <= 60
            and claims["extracted"] is True
            and claims["extracted_value"] > 0
        )
        if policy == "aggressive":
            passed = passed and (
                claims["valid_hits"] >= 5
                and claims["kills"] >= 1
                and claims["cache_looted"] >= 1
            )
        elif policy == "defensive":
            passed = passed and claims["attack_decisions"] <= 5
        elif policy == "explorer":
            passed = passed and claims["unique_route_cells"] >= 8
        if not passed:
            raise ValueError(f"{policy} showcase is not representative")
        media_checks[policy] = {
            "duration_seconds": duration,
            "claims": claims,
            "passed": True,
        }
    result = {
        "schema_version": 1,
        "status": "PASS",
        "release_manifest": str(release_path.relative_to(root)),
        "release_manifest_sha256": sha256_file(release_path),
        "release_sha256": release["release_sha256"],
        "official_receipt": str(receipt_path.relative_to(root)),
        "official_receipt_sha256": sha256_file(receipt_path),
        "showcase_manifest": str(showcase_path.relative_to(root)),
        "showcase_manifest_sha256": sha256_file(showcase_path),
        "media_checks": media_checks,
        "test_cases_accessed_only_by_official_runner": True,
    }
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
