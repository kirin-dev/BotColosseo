from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

import torch
import yaml

from botcolosseo.agents.hybrid_config import load_hybrid_policy_config
from botcolosseo.agents.hybrid_policy import build_hybrid_evaluation_policy
from botcolosseo.agents.league_opponents import (
    CheckpointOpponentPolicy,
    OpponentSpec,
    sha256_file,
)
from botcolosseo.evaluation.hybrid_showcase import load_hybrid_showcase_config
from botcolosseo.evaluation.showcase import canonical_json

_POLICY_IDS = ("strong_base", "aggressive", "defensive", "explorer")
_POLICY_KINDS = ("checkpoint", "checkpoint", "hybrid_config", "hybrid_config")
_POLICY_LABELS = (
    "Strong Base (learned)",
    "Aggressive (learned style)",
    "Defensive (hybrid governor)",
    "Explorer (hybrid governor)",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def build_hybrid_release(
    *,
    root: Path,
    config_path: Path,
    showcase_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    root = root.resolve()
    config = load_hybrid_showcase_config(config_path, root=root)
    showcase_manifest_path = showcase_manifest_path.resolve()
    showcase = json.loads(showcase_manifest_path.read_text(encoding="utf-8"))
    expected_artifacts = {
        row.policy_id: row.expected_sha256 for row in config.policies
    }
    if (
        showcase.get("stage") != "hybrid_product_showcase"
        or showcase.get("publication") is not True
        or showcase.get("test_cases_accessed") is not False
        or showcase.get("config_sha256") != config.config_sha256
        or showcase.get("policy_artifact_sha256") != expected_artifacts
    ):
        raise ValueError("Hybrid release showcase evidence is invalid")
    scenario_hash = showcase.get("scenario_hash")
    if not isinstance(scenario_hash, str) or _SHA256.fullmatch(scenario_hash) is None:
        raise ValueError("Hybrid release scenario hash is invalid")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite a hybrid release")

    hybrid_sources = {}
    for row in config.policies:
        if row.kind == "checkpoint":
            spec = OpponentSpec(
                opponent_id=row.policy_id,
                kind="checkpoint",
                checkpoint=str(row.artifact),
                checkpoint_sha256=row.expected_sha256,
                scenario_hash=scenario_hash,
                selection_evidence=f"hybrid-release:{row.policy_id}",
            )
            CheckpointOpponentPolicy.load(spec, device=torch.device("cpu"))
        else:
            hybrid = load_hybrid_policy_config(row.artifact, root=root)
            if hybrid.style != row.policy_id:
                raise ValueError("Hybrid release governor style does not match")
            build_hybrid_evaluation_policy(hybrid, device=torch.device("cpu"))
            hybrid_sources[row.policy_id] = row.artifact

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-",
        dir=output_dir.parent,
    ) as directory:
        staging = Path(directory) / "package"
        checkpoints = staging / "checkpoints"
        governors = staging / "governors"
        evidence_dir = staging / "evidence"
        checkpoints.mkdir(parents=True)
        governors.mkdir()
        evidence_dir.mkdir()
        policy_rows = []
        for row in config.policies:
            if row.kind == "checkpoint":
                target = checkpoints / f"{row.policy_id.replace('_', '-')}.pt"
                shutil.copyfile(row.artifact, target)
                if sha256_file(target) != row.expected_sha256:
                    raise RuntimeError("Hybrid release checkpoint copy drifted")
                policy_rows.append(
                    {
                        "policy_id": row.policy_id,
                        "label": row.label,
                        "kind": row.kind,
                        "path": target.relative_to(staging).as_posix(),
                        "sha256": row.expected_sha256,
                        "bytes": target.stat().st_size,
                    }
                )
                continue
            source = hybrid_sources[row.policy_id]
            payload = yaml.safe_load(source.read_text(encoding="utf-8"))
            payload["base_checkpoint"] = "checkpoints/strong-base.pt"
            target = governors / f"{row.policy_id}.yaml"
            target.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            portable = load_hybrid_policy_config(target, root=staging)
            if portable.style != row.policy_id:
                raise RuntimeError("Portable hybrid governor validation failed")
            policy_rows.append(
                {
                    "policy_id": row.policy_id,
                    "label": row.label,
                    "kind": row.kind,
                    "path": target.relative_to(staging).as_posix(),
                    "sha256": sha256_file(target),
                    "bytes": target.stat().st_size,
                    "source_config_sha256": row.expected_sha256,
                    "base_checkpoint_sha256": portable.base_checkpoint_sha256,
                }
            )
        evidence_rows = []
        for row in config.evidence:
            target = evidence_dir / f"{row.style}-formal-summary.json"
            shutil.copyfile(row.summary, target)
            if sha256_file(target) != row.expected_sha256:
                raise RuntimeError("Hybrid release evidence copy drifted")
            evidence_rows.append(
                {
                    "style": row.style,
                    "path": target.relative_to(staging).as_posix(),
                    "sha256": row.expected_sha256,
                }
            )
        showcase_target = evidence_dir / "showcase-manifest.json"
        shutil.copyfile(showcase_manifest_path, showcase_target)
        manifest = {
            "schema_version": 1,
            "stage": "hybrid-product-release",
            "scenario_hash": scenario_hash,
            "showcase_manifest_sha256": sha256_file(showcase_target),
            "policies": policy_rows,
            "evidence": evidence_rows,
            "total_bytes": sum(int(row["bytes"]) for row in policy_rows),
            "fair_observation_loader_verified": True,
            "portable_governor_configs_verified": True,
            "distribution": "github-release",
            "test_cases_accessed": False,
        }
        (staging / "manifest.json").write_bytes(canonical_json(manifest))
        staging.replace(output_dir)
    return manifest


def audit_hybrid_release(package_dir: Path) -> dict[str, object]:
    package_dir = package_dir.resolve()
    manifest = _json_object(package_dir / "manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("stage") != "hybrid-product-release"
        or manifest.get("distribution") != "github-release"
        or manifest.get("test_cases_accessed") is not False
        or manifest.get("fair_observation_loader_verified") is not True
        or manifest.get("portable_governor_configs_verified") is not True
    ):
        raise ValueError("Hybrid release manifest identity is invalid")
    scenario_hash = manifest.get("scenario_hash")
    if not isinstance(scenario_hash, str) or _SHA256.fullmatch(scenario_hash) is None:
        raise ValueError("Hybrid release scenario hash is invalid")

    policies = manifest.get("policies")
    if (
        not isinstance(policies, list)
        or tuple(row.get("policy_id") for row in policies if isinstance(row, dict))
        != _POLICY_IDS
        or tuple(row.get("kind") for row in policies if isinstance(row, dict))
        != _POLICY_KINDS
        or tuple(row.get("label") for row in policies if isinstance(row, dict))
        != _POLICY_LABELS
        or len(policies) != len(_POLICY_IDS)
    ):
        raise ValueError("Hybrid release policy order is invalid")
    total_bytes = 0
    for row in policies:
        if not isinstance(row, dict):
            raise ValueError("Hybrid release policy row is invalid")
        policy_id = str(row["policy_id"])
        path = _release_member(package_dir, row.get("path"))
        digest = row.get("sha256")
        if not isinstance(digest, str) or sha256_file(path) != digest:
            raise ValueError(f"Hybrid release policy hash drifted: {policy_id}")
        if path.stat().st_size != row.get("bytes"):
            raise ValueError(f"Hybrid release policy size drifted: {policy_id}")
        total_bytes += path.stat().st_size
        kind = row.get("kind")
        if kind == "checkpoint":
            spec = OpponentSpec(
                opponent_id=policy_id,
                kind="checkpoint",
                checkpoint=str(path),
                checkpoint_sha256=digest,
                scenario_hash=scenario_hash,
                selection_evidence=f"hybrid-release-audit:{policy_id}",
            )
            CheckpointOpponentPolicy.load(spec, device=torch.device("cpu"))
        elif kind == "hybrid_config":
            config = load_hybrid_policy_config(path, root=package_dir)
            if config.style != policy_id or config.scenario_hash != scenario_hash:
                raise ValueError(f"Hybrid release governor identity drifted: {policy_id}")
            build_hybrid_evaluation_policy(config, device=torch.device("cpu"))
        else:
            raise ValueError(f"Hybrid release policy kind is invalid: {policy_id}")
    if total_bytes != manifest.get("total_bytes"):
        raise ValueError("Hybrid release total bytes drifted")

    evidence = manifest.get("evidence")
    if (
        not isinstance(evidence, list)
        or tuple(row.get("style") for row in evidence if isinstance(row, dict))
        != ("aggressive", "defensive", "explorer")
        or len(evidence) != 3
    ):
        raise ValueError("Hybrid release evidence list is invalid")
    for row in evidence:
        if not isinstance(row, dict):
            raise ValueError("Hybrid release evidence row is invalid")
        path = _release_member(package_dir, row.get("path"))
        if sha256_file(path) != row.get("sha256"):
            raise ValueError(f"Hybrid release evidence hash drifted: {row.get('style')}")
    showcase = package_dir / "evidence" / "showcase-manifest.json"
    if sha256_file(showcase) != manifest.get("showcase_manifest_sha256"):
        raise ValueError("Hybrid release showcase manifest hash drifted")
    return manifest


def _release_member(package_dir: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError("Hybrid release member path is invalid")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("Hybrid release member path must be relative")
    path = (package_dir / relative).resolve()
    if not path.is_relative_to(package_dir) or not path.is_file():
        raise ValueError("Hybrid release member path escapes the package")
    return path


def _json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload
