import hashlib
import json
from pathlib import Path

from botcolosseo.evaluation.defensive_evidence_audit import audit_defensive_evidence


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _evidence(root: Path) -> None:
    base_hash = "a" * 64
    scenario = "scenario"
    data_dir = root / "data/generated/m5/defensive"
    shards = []
    for index in range(5):
        shard = data_dir / f"train-{index:05d}.npz"
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_bytes(f"shard-{index}".encode())
        shards.append(
            {
                "file": shard.name,
                "sha256": _sha(shard),
                "transitions": 10_000,
            }
        )
    data_manifest = data_dir / "train-manifest.json"
    _write(
        data_manifest,
        {
            "passed": True,
            "style": "defensive",
            "split": "train",
            "transitions": 50_000,
            "gate": {
                "complete": True,
                "risk_transitions": True,
                "denial_recovery_windows": True,
                "opponent_coverage": True,
                "side_coverage": True,
                "no_risk_balance": True,
            },
            "shards": shards,
            "base_checkpoint_sha256": base_hash,
            "scenario_hash": scenario,
            "test_cases_accessed": False,
        },
    )
    run_dir = root / "runs/m5/defensive-distillation"
    distilled = run_dir / "style-pretrained.pt"
    neutral = run_dir / "neutral.pt"
    distilled.parent.mkdir(parents=True, exist_ok=True)
    distilled.write_bytes(b"distilled")
    neutral.write_bytes(b"neutral")
    _write(
        run_dir / "summary.json",
        {
            "passed": True,
            "completed": True,
            "style": "defensive",
            "frozen_base": True,
            "base_checkpoint_sha256": base_hash,
            "data_manifest_sha256": _sha(data_manifest),
            "data_gate_passed": True,
            "data_waiver": None,
            "data_waiver_applied": False,
            "data_waiver_sha256": None,
            "offline_evaluation": {"passed": True},
            "checkpoint": str(distilled),
            "checkpoint_sha256": _sha(distilled),
            "neutral_checkpoint": str(neutral),
            "neutral_checkpoint_sha256": _sha(neutral),
            "scenario_hash": scenario,
            "test_cases_accessed": False,
        },
    )
    selected_checkpoint = root / "runs/m5/defensive-interpolation/alpha-050.pt"
    selected_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    selected_checkpoint.write_bytes(b"selected")
    selected_hash = _sha(selected_checkpoint)
    _write(
        root / "reports/m5/defensive/selection.json",
        {
            "passed": True,
            "grid": [0.25, 0.5, 0.75],
            "selected": {
                "alpha": 0.5,
                "eligible": True,
                "protective_presence_estimator": "paired_cluster_bootstrap_pooled_ratio_v1",
                "checkpoint": str(selected_checkpoint),
                "checkpoint_sha256": selected_hash,
            },
            "test_cases_accessed": False,
        },
    )
    formal = root / "reports/m5/defensive/formal"
    episodes = formal / "episodes.jsonl"
    episodes.parent.mkdir(parents=True, exist_ok=True)
    episodes.write_text(
        "".join(
            json.dumps(
                {
                    "policy": "defensive" if index % 2 else "strong_base",
                    "opponent": f"opponent-{index % 5}",
                    "pair_index": index // 4,
                    "learner_side": "host" if (index // 2) % 2 == 0 else "opponent",
                    "row": index,
                },
                sort_keys=True,
            )
            + "\n"
            for index in range(200)
        ),
        encoding="utf-8",
    )
    summary = formal / "summary.json"
    _write(
        summary,
        {
            "passed": True,
            "complete": True,
            "episodes": 200,
            "expected_episodes": 200,
            "gates": {"complete": True, "retention": True},
            "protective_presence_estimator": "paired_cluster_bootstrap_pooled_ratio_v1",
            "checkpoint_sha256": {
                "strong_base": base_hash,
                "defensive": selected_hash,
            },
            "test_cases_accessed": False,
        },
    )
    _write(
        formal / "manifest.json",
        {
            "passed": True,
            "episodes": 200,
            "episodes_sha256": _sha(episodes),
            "summary_sha256": _sha(summary),
            "scenario_hash": scenario,
            "test_cases_accessed": False,
        },
    )


def test_defensive_evidence_audit_accepts_complete_hash_chain(tmp_path: Path) -> None:
    _evidence(tmp_path)

    result = audit_defensive_evidence(tmp_path)

    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["selected_alpha"] == 0.5


def test_defensive_evidence_audit_reports_selected_checkpoint_tampering(
    tmp_path: Path,
) -> None:
    _evidence(tmp_path)
    checkpoint = tmp_path / "runs/m5/defensive-interpolation/alpha-050.pt"
    checkpoint.write_bytes(b"tampered")

    result = audit_defensive_evidence(tmp_path)

    assert result["passed"] is False
    assert result["checks"]["selection_gate"] is False
    assert result["checks"]["formal_gate"] is True
