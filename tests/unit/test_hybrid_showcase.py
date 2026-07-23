from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.evaluation.hybrid_showcase import load_hybrid_showcase_config


def _write(root: Path, relative: str, content: bytes = b"artifact") -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return sha256_file(path)


def _payload(root: Path) -> dict[str, object]:
    case_hash = _write(root, "configs/showcase/cases.json")
    policy_paths = (
        "runs/base.pt",
        "runs/aggressive.pt",
        "configs/defensive.yaml",
        "configs/explorer.yaml",
    )
    policy_hashes = [_write(root, path) for path in policy_paths]
    evidence_paths = (
        "reports/aggressive.json",
        "reports/defensive.json",
        "reports/explorer.json",
    )
    evidence_hashes = [_write(root, path) for path in evidence_paths]
    return {
        "schema_version": 1,
        "stage": "hybrid_product",
        "publication": True,
        "split": "validation",
        "test_cases_accessed": False,
        "cases": {
            "manifest": "configs/showcase/cases.json",
            "expected_sha256": case_hash,
            "selected_case_id": "aggressive_script:252:host",
        },
        "policies": [
            {
                "id": policy_id,
                "label": label,
                "kind": kind,
                "artifact": artifact,
                "expected_sha256": digest,
            }
            for policy_id, label, kind, artifact, digest in zip(
                ("strong_base", "aggressive", "defensive", "explorer"),
                (
                    "Strong Base (learned)",
                    "Aggressive (learned style)",
                    "Defensive (hybrid governor)",
                    "Explorer (hybrid governor)",
                ),
                ("checkpoint", "checkpoint", "hybrid_config", "hybrid_config"),
                policy_paths,
                policy_hashes,
                strict=True,
            )
        ],
        "evidence": [
            {
                "style": style,
                "summary": path,
                "expected_sha256": digest,
            }
            for style, path, digest in zip(
                ("aggressive", "defensive", "explorer"),
                evidence_paths,
                evidence_hashes,
                strict=True,
            )
        ],
        "render": {
            "fps": 10,
            "gif_seconds": 18,
            "gif_max_bytes": 10_000_000,
            "max_decisions": 525,
            "output_dir": "docs/assets/showcase",
            "evidence_dir": "reports/showcase/hybrid-product",
        },
    }


def test_hybrid_showcase_config_discloses_policy_kinds_and_binds_hashes(
    tmp_path: Path,
) -> None:
    payload = _payload(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = load_hybrid_showcase_config(path, root=tmp_path)

    assert tuple(row.policy_id for row in config.policies) == (
        "strong_base",
        "aggressive",
        "defensive",
        "explorer",
    )
    assert tuple(row.kind for row in config.policies) == (
        "checkpoint",
        "checkpoint",
        "hybrid_config",
        "hybrid_config",
    )
    assert config.selected_case_id == "aggressive_script:252:host"


def test_hybrid_showcase_config_rejects_test_access_and_hash_drift(
    tmp_path: Path,
) -> None:
    payload = _payload(tmp_path)
    path = tmp_path / "config.yaml"
    payload["test_cases_accessed"] = True
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="publication boundary"):
        load_hybrid_showcase_config(path, root=tmp_path)

    payload["test_cases_accessed"] = False
    payload["policies"][0]["expected_sha256"] = "0" * 64  # type: ignore[index]
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="hash does not match"):
        load_hybrid_showcase_config(path, root=tmp_path)
