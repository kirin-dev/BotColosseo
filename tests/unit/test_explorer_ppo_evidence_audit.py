import hashlib
import json
from pathlib import Path

from botcolosseo.evaluation.explorer import ROUTE_ENTROPY_ESTIMATOR
from botcolosseo.evaluation.explorer_ppo_evidence_audit import (
    audit_explorer_ppo_evidence,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _evidence(root: Path) -> None:
    scenario = "scenario"
    base_hash = "a" * 64
    interpolation = root / "runs/m5/explorer-interpolation"
    warm = interpolation / "alpha-025.pt"
    warm.parent.mkdir(parents=True)
    warm.write_bytes(b"warm")
    _write(
        interpolation / "alpha-025.json",
        {
            "style": "explorer",
            "alpha": 0.25,
            "checkpoint_sha256": _sha(warm),
            "scenario_hash": scenario,
            "test_cases_accessed": False,
        },
    )
    run = root / "runs/m5/explorer-ppo-main"
    candidate = run / "candidate-0200000.pt"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"candidate")
    candidate_hash = _sha(candidate)
    _write(
        run / "summary.json",
        {
            "completed": True,
            "environment_steps": 200_000,
            "style": "explorer",
            "style_warm_start_sha256": _sha(warm),
            "candidate_checkpoints": [
                {"environment_steps": 200_000, "sha256": candidate_hash}
            ],
            "base_checkpoint_sha256": base_hash,
            "scenario_hash": scenario,
            "test_cases_accessed": False,
        },
    )
    smoke = root / "reports/m5/explorer/ppo-repair/smoke"
    smoke_summary = smoke / "summary.json"
    _write(
        smoke_summary,
        {
            "passed": True,
            "route_entropy_estimator": ROUTE_ENTROPY_ESTIMATOR,
            "checkpoint_sha256": {"explorer": candidate_hash},
            "test_cases_accessed": False,
        },
    )
    _write(
        smoke / "manifest.json",
        {
            "passed": True,
            "episodes": 20,
            "summary_sha256": _sha(smoke_summary),
            "scenario_hash": scenario,
            "test_cases_accessed": False,
        },
    )
    formal = root / "reports/m5/explorer/ppo-repair/formal"
    episodes = formal / "episodes.jsonl"
    episodes.parent.mkdir(parents=True)
    episodes.write_text(
        "".join(
            json.dumps(
                {
                    "policy": "explorer" if index % 2 else "strong_base",
                    "opponent": f"opponent-{index % 5}",
                    "pair_index": index // 4,
                    "learner_side": "host" if (index // 2) % 2 == 0 else "opponent",
                },
                sort_keys=True,
            )
            + "\n"
            for index in range(200)
        ),
        encoding="utf-8",
    )
    formal_summary = formal / "summary.json"
    _write(
        formal_summary,
        {
            "passed": True,
            "complete": True,
            "gates": {"complete": True, "style": True},
            "route_entropy_estimator": ROUTE_ENTROPY_ESTIMATOR,
            "checkpoint_sha256": {
                "explorer": candidate_hash,
                "strong_base": base_hash,
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
            "summary_sha256": _sha(formal_summary),
            "scenario_hash": scenario,
            "test_cases_accessed": False,
        },
    )


def test_explorer_ppo_audit_accepts_complete_hash_chain(tmp_path: Path) -> None:
    _evidence(tmp_path)

    result = audit_explorer_ppo_evidence(tmp_path)

    assert result["passed"] is True
    assert all(result["checks"].values())


def test_explorer_ppo_audit_rejects_candidate_tampering(tmp_path: Path) -> None:
    _evidence(tmp_path)
    candidate = tmp_path / "runs/m5/explorer-ppo-main/candidate-0200000.pt"
    candidate.write_bytes(b"tampered")

    result = audit_explorer_ppo_evidence(tmp_path)

    assert result["passed"] is False
    assert result["checks"]["training_gate"] is False
    assert result["checks"]["formal_gate"] is False
