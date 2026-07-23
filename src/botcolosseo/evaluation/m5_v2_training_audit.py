from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import torch

from botcolosseo.agents.league_opponents import (
    CheckpointOpponentPolicy,
    OpponentSpec,
    sha256_file,
)

_FINITE_TRAIN_FIELDS = (
    "total_loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "approximate_kl",
    "pre_clip_grad_norm",
    "post_clip_grad_norm",
    "style_base_kl",
    "auxiliary_loss",
    "teacher_agreement",
)


def audit_m5_v2_training(
    run_dir: Path,
    *,
    style: str,
    expected_steps: int,
) -> dict[str, object]:
    if style not in ("defensive", "explorer") or expected_steps <= 0:
        raise ValueError("Invalid M5 V2 audit request")
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not summary_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("M5 V2 training evidence is incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = Path(str(summary.get("checkpoint", ""))).resolve()
    if (
        summary.get("style") != style
        or summary.get("environment_steps") != expected_steps
        or summary.get("test_cases_accessed") is not False
        or summary.get("teacher") is None
        or not checkpoint.is_file()
        or summary.get("checkpoint_sha256") != sha256_file(checkpoint)
    ):
        raise ValueError("M5 V2 summary identity is invalid")

    train_records = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") == "train":
            train_records.append(record)
    if not train_records:
        raise ValueError("M5 V2 evidence contains no optimizer updates")
    for record in train_records:
        if any(
            isinstance(record.get(name), bool)
            or not isinstance(record.get(name), (int, float))
            or not math.isfinite(float(record[name]))
            for name in _FINITE_TRAIN_FIELDS
        ):
            raise ValueError("M5 V2 training metric is missing or non-finite")

    supervision = summary.get("supervision_counts")
    if not isinstance(supervision, dict) or int(supervision.get("tokens", 0)) <= 0:
        raise ValueError("M5 V2 supervision mask is empty")
    if style == "explorer":
        if any(int(supervision.get(f"mode:{mode}", 0)) <= 0 for mode in range(3)):
            raise ValueError("Explorer V2 did not expose every route mode")
        rewards = summary.get("style_reward_components")
        if not isinstance(rewards, dict) or any(
            not any(name.startswith(f"mode:{mode}:") for name in rewards)
            for mode in range(3)
        ):
            raise ValueError("Explorer V2 has incomplete per-mode reward evidence")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("M5 V2 checkpoint has no model state")
    if style == "explorer" and any(
        not any(name.startswith(f"adapters.{mode}.") for name in state)
        or not any(name.startswith(f"policies.{mode}.") for name in state)
        for mode in range(3)
    ):
        raise ValueError("Explorer V2 checkpoint is missing a route branch")
    scenario_hash = str(summary["scenario_hash"])
    spec = OpponentSpec(
        opponent_id=f"{style}-v2-audit",
        kind="checkpoint",
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        scenario_hash=scenario_hash,
        selection_evidence=str(summary_path),
    )
    policy = CheckpointOpponentPolicy.load(spec, device=torch.device("cpu"))
    if list(inspect.signature(policy.act).parameters) != ["observation"]:
        raise ValueError("Published policy interface is not public-observation-only")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "environment_steps": expected_steps,
        "passed": True,
        "style": style,
        "supervision_counts": supervision,
        "test_cases_accessed": False,
        "train_updates_audited": len(train_records),
    }
