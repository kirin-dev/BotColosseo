from __future__ import annotations

import json
from pathlib import Path

import yaml

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.evaluation.m6_release import M6_POLICIES
from botcolosseo.evaluation.showcase import (
    M6ShowcaseMetricEvidence,
    load_metric_evidence,
)

M6_POLICY_INPUTS = {
    "strong_base": (
        "Strong Base",
        "runs/m3/league-full/candidate-boundary-0200000.pt",
    ),
    "aggressive": (
        "Aggressive",
        "runs/m4/aggressive-interpolation/alpha-025.pt",
    ),
    "defensive": (
        "Defensive",
        "runs/m5/defensive-ppo-main/candidate-0200000.pt",
    ),
    "explorer": (
        "Explorer",
        "runs/m5/explorer-ppo-main/candidate-0200000.pt",
    ),
}


def build_m6_showcase_config(
    *,
    root: Path,
    metrics_path: Path,
) -> dict[str, object]:
    root = root.expanduser().resolve()
    metrics_path = metrics_path.expanduser().resolve()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("checkpoint_sha256"), dict
    ):
        raise ValueError("M6 showcase metrics are invalid")
    evidence = load_metric_evidence(
        metrics_path,
        expected_stage="m6",
        expected_hashes=payload["checkpoint_sha256"],
    )
    if not isinstance(evidence, M6ShowcaseMetricEvidence):
        raise ValueError("M6 showcase config requires M6 metric evidence")
    policies = []
    for policy_id in M6_POLICIES:
        label, relative = M6_POLICY_INPUTS[policy_id]
        checkpoint = root / relative
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        digest = sha256_file(checkpoint)
        if digest != evidence.checkpoint_sha256[policy_id]:
            raise ValueError(f"M6 showcase checkpoint hash drift: {policy_id}")
        policies.append(
            {
                "id": policy_id,
                "label": label,
                "checkpoint": relative,
                "expected_sha256": digest,
            }
        )
    try:
        metrics_relative = metrics_path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("M6 metrics must be inside the repository") from error
    return {
        "schema_version": 1,
        "stage": "m6",
        "publication": True,
        "split": "validation",
        "cases": "configs/showcase/m4-validation.json",
        "metrics": metrics_relative,
        "policies": policies,
        "render": {
            "fps": 10,
            "gif_seconds": 18,
            "gif_max_bytes": 10_000_000,
            "max_decisions": 525,
            "output_dir": "docs/assets/showcase",
        },
        "evidence_dir": "reports/showcase/m6",
    }


def write_m6_showcase_config(
    payload: dict[str, object],
    output: Path,
) -> Path:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite the M6 showcase config")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return output
