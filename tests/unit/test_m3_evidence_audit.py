import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.cli.evaluate_m3 import (
    append_episode_row,
    write_m3_evidence,
)
from botcolosseo.evaluation.m3 import M3EpisodeRecord
from botcolosseo.evaluation.m3_evidence_audit import audit_m3_evidence


@dataclass(frozen=True)
class StubSummary:
    episodes: int = 1
    expected_episodes: int = 1
    official: bool = True
    complete: bool = True
    passed: bool = True
    pool_size: int = 8

    def to_dict(self) -> dict[str, object]:
        return {
            "episodes": self.episodes,
            "expected_episodes": self.expected_episodes,
            "official": self.official,
            "complete": self.complete,
            "passed": self.passed,
            "pool_size": self.pool_size,
        }


def _row() -> M3EpisodeRecord:
    return M3EpisodeRecord(
        policy="strong_base",
        category="script",
        split="test",
        opponent="objective_first",
        pair_index=1,
        seed=7,
        learner_side="host",
        outcome="win",
        objective_completed=True,
        goal_reached=True,
        pickup_completed=True,
        return_completed=True,
        valid_hit=True,
        disengage_success=True,
        learner_score=1,
        opponent_score=0,
        actual_core_x=0.0,
        actual_core_y=0.0,
        decisions=10,
        terminated=True,
        truncated=False,
        peer_tic_lag_max=0,
        protocol_inconsistent=False,
        action_tic_inconsistent=False,
        score_event_inconsistent=False,
        fairness_schema_inconsistent=False,
        scenario_hash="scenario",
        environment_attempts=1,
    )


def _evidence(tmp_path: Path) -> tuple[Path, StubSummary]:
    report_dir = tmp_path / "official"
    report_dir.mkdir(parents=True)
    identity = {
        "official": True,
        "historical_policy_ids": [f"historical-{index:02d}" for index in range(8)],
        "scenario_hash": "scenario",
        "selected_checkpoint_sha256": "a" * 64,
        "pool_manifest_sha256": "b" * 64,
        "m2_baseline_sha256": "c" * 64,
    }
    (report_dir / "run-identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    append_episode_row(report_dir / "episodes.jsonl", _row())
    summary = StubSummary()
    write_m3_evidence(report_dir, summary=summary, run_identity=identity)
    return report_dir, summary


def test_audit_recomputes_summary_from_hash_bound_raw_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_dir, summary = _evidence(tmp_path)
    calls = []

    def recompute(rows, **kwargs):
        calls.append((tuple(rows), kwargs))
        return summary

    monkeypatch.setattr(
        "botcolosseo.evaluation.m3_evidence_audit.evaluate_m3_records", recompute
    )

    result = audit_m3_evidence(report_dir)

    assert result["passed"] is True
    assert len(calls) == 1
    assert calls[0][0] == (_row(),)
    assert calls[0][1]["expected_scenario_hash"] == "scenario"


def test_audit_rejects_missing_or_tampered_artifacts_even_with_updated_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_dir, summary = _evidence(tmp_path)
    monkeypatch.setattr(
        "botcolosseo.evaluation.m3_evidence_audit.evaluate_m3_records",
        lambda rows, **kwargs: summary,
    )
    stored = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    stored["passed"] = False
    (report_dir / "summary.json").write_text(
        json.dumps(stored, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = report_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["summary_sha256"] = sha256_file(report_dir / "summary.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="raw-row recomputation"):
        audit_m3_evidence(report_dir)

    (report_dir / "episodes.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="episodes"):
        audit_m3_evidence(report_dir)
