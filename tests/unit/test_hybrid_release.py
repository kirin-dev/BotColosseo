from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import botcolosseo.evaluation.hybrid_release as release


def test_hybrid_release_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "release"
    output.mkdir()

    with pytest.raises(FileExistsError, match="overwrite"):
        release.build_hybrid_release(
            root=Path.cwd(),
            config_path=Path("configs/showcase/hybrid-product.yaml"),
            showcase_manifest_path=Path(
                "reports/showcase/hybrid-product/manifest.json"
            ),
            difficulty_summary_path=Path(
                "reports/m5/difficulty/hybrid-all-style-summary.json"
            ),
            m6_metrics_path=Path("reports/m6/hybrid-product-metrics.json"),
            difficulty_plot_path=Path(
                "docs/assets/showcase/m5-hybrid-all-style-difficulty.png"
            ),
            output_dir=output,
        )


def test_hybrid_release_rejects_showcase_identity_drift(tmp_path: Path) -> None:
    source = Path("reports/showcase/hybrid-product/manifest.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["config_sha256"] = "0" * 64
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="showcase evidence"):
        release.build_hybrid_release(
            root=Path.cwd(),
            config_path=Path("configs/showcase/hybrid-product.yaml"),
            showcase_manifest_path=manifest,
            difficulty_summary_path=Path(
                "reports/m5/difficulty/hybrid-all-style-summary.json"
            ),
            m6_metrics_path=Path("reports/m6/hybrid-product-metrics.json"),
            difficulty_plot_path=Path(
                "docs/assets/showcase/m5-hybrid-all-style-difficulty.png"
            ),
            output_dir=tmp_path / "release",
        )


def test_hybrid_release_audit_rejects_policy_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "release"
    checkpoints = package / "checkpoints"
    governors = package / "governors"
    evidence = package / "evidence"
    checkpoints.mkdir(parents=True)
    governors.mkdir()
    evidence.mkdir()
    scenario_hash = "1" * 64
    policy_specs = (
        (
            "strong_base",
            "Strong Base (learned)",
            "checkpoint",
            checkpoints / "strong-base.pt",
        ),
        (
            "aggressive",
            "Aggressive (learned style)",
            "checkpoint",
            checkpoints / "aggressive.pt",
        ),
        (
            "defensive",
            "Defensive (hybrid governor)",
            "hybrid_config",
            governors / "defensive.yaml",
        ),
        (
            "explorer",
            "Explorer (hybrid governor)",
            "hybrid_config",
            governors / "explorer.yaml",
        ),
    )
    policy_rows = []
    for policy_id, label, kind, path in policy_specs:
        path.write_bytes(policy_id.encode())
        policy_rows.append(
            {
                "policy_id": policy_id,
                "label": label,
                "kind": kind,
                "path": path.relative_to(package).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    evidence_rows = []
    for style in ("aggressive", "defensive", "explorer"):
        path = evidence / f"{style}-formal-summary.json"
        path.write_text("{}", encoding="utf-8")
        evidence_rows.append(
            {
                "style": style,
                "path": path.relative_to(package).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    showcase = evidence / "showcase-manifest.json"
    showcase.write_text("{}", encoding="utf-8")
    difficulty = evidence / "difficulty-summary.json"
    difficulty.write_text("{}", encoding="utf-8")
    metrics = evidence / "m6-product-metrics.json"
    metrics.write_text("{}", encoding="utf-8")
    assets = package / "assets"
    assets.mkdir()
    plot = assets / "style-difficulty.png"
    plot.write_bytes(b"\x89PNG\r\n\x1a\n")
    manifest = {
        "schema_version": 2,
        "stage": "hybrid-product-release",
        "distribution": "github-release",
        "test_cases_accessed": False,
        "fair_observation_loader_verified": True,
        "portable_governor_configs_verified": True,
        "scenario_hash": scenario_hash,
        "policies": policy_rows,
        "evidence": evidence_rows,
        "showcase_manifest_sha256": hashlib.sha256(
            showcase.read_bytes()
        ).hexdigest(),
        "difficulty_summary_sha256": hashlib.sha256(
            difficulty.read_bytes()
        ).hexdigest(),
        "m6_metrics_sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
        "difficulty_plot_sha256": hashlib.sha256(plot.read_bytes()).hexdigest(),
        "total_bytes": sum(path.stat().st_size for _, _, _, path in policy_specs),
    }
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(release.CheckpointOpponentPolicy, "load", lambda *a, **k: None)
    monkeypatch.setattr(
        release,
        "load_hybrid_policy_config",
        lambda path, root: SimpleNamespace(
            style=path.stem,
            scenario_hash=scenario_hash,
        ),
    )
    monkeypatch.setattr(
        release,
        "build_hybrid_evaluation_policy",
        lambda *a, **k: None,
    )
    checkpoint = package / "checkpoints" / "aggressive.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"drift")

    with pytest.raises(ValueError, match="policy hash drifted: aggressive"):
        release.audit_hybrid_release(package)
