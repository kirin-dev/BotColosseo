from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch

from botcolosseo.agents.league_opponents import (
    CheckpointOpponentPolicy,
    OpponentSpec,
    sha256_file,
)
from botcolosseo.evaluation.m6_release import M6_POLICIES
from botcolosseo.evaluation.showcase import (
    M6ShowcaseMetricEvidence,
    canonical_json,
    load_metric_evidence,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def build_checkpoint_release(
    *,
    metrics_path: Path,
    sources: Mapping[str, Path],
    scenario_hash: str,
    output_dir: Path,
) -> dict[str, object]:
    if tuple(sources) != M6_POLICIES:
        raise ValueError("Checkpoint release requires the frozen four-policy order")
    if _SHA256.fullmatch(scenario_hash) is None:
        raise ValueError("Checkpoint release scenario hash is invalid")
    metrics_path = metrics_path.expanduser().resolve()
    payload = _json(metrics_path)
    expected_hashes = payload.get("checkpoint_sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("M6 metrics do not contain checkpoint hashes")
    evidence = load_metric_evidence(
        metrics_path,
        expected_stage="m6",
        expected_hashes=expected_hashes,
    )
    if not isinstance(evidence, M6ShowcaseMetricEvidence):
        raise ValueError("Checkpoint release requires M6 metric evidence")

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite a checkpoint release")
    verified_sources: list[tuple[str, Path, str]] = []
    for policy_id in M6_POLICIES:
        source = sources[policy_id].expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = sha256_file(source)
        if digest != evidence.checkpoint_sha256[policy_id]:
            raise ValueError(f"Release checkpoint hash does not match: {policy_id}")
        _verify_checkpoint(
            source,
            policy_id=policy_id,
            checkpoint_sha256=digest,
            scenario_hash=scenario_hash,
        )
        verified_sources.append((policy_id, source, digest))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary) / "package"
        checkpoints_dir = staging / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        rows: list[dict[str, object]] = []
        for policy_id, source, digest in verified_sources:
            target = checkpoints_dir / f"{policy_id.replace('_', '-')}.pt"
            shutil.copyfile(source, target)
            if sha256_file(target) != digest:
                raise RuntimeError("Checkpoint release copy changed its content hash")
            rows.append(
                {
                    "policy_id": policy_id,
                    "path": target.relative_to(staging).as_posix(),
                    "sha256": digest,
                    "bytes": target.stat().st_size,
                    "source_checkpoint_preserved": True,
                }
            )
        manifest = {
            "schema_version": 1,
            "stage": "m6-checkpoint-release",
            "scenario_hash": scenario_hash,
            "metrics_sha256": sha256_file(metrics_path),
            "policies": rows,
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "fair_observation_loader_verified": True,
            "optimizer_state_stripped": False,
            "distribution": "github-release",
            "test_cases_accessed": False,
        }
        (staging / "manifest.json").write_bytes(canonical_json(manifest))
        staging.replace(output_dir)
    return manifest


def _verify_checkpoint(
    path: Path,
    *,
    policy_id: str,
    checkpoint_sha256: str,
    scenario_hash: str,
) -> None:
    spec = OpponentSpec(
        opponent_id=policy_id,
        kind="checkpoint",
        checkpoint=str(path),
        checkpoint_sha256=checkpoint_sha256,
        scenario_hash=scenario_hash,
        selection_evidence=f"m6-release:{policy_id}",
    )
    CheckpointOpponentPolicy.load(spec, device=torch.device("cpu"))


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload
