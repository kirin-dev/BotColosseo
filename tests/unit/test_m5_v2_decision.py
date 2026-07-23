import json
from pathlib import Path

import pytest

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.evaluation.m5_v2_decision import decide_m5_v2_candidate


def _evidence(tmp_path: Path, *, retention: bool, primary: float) -> dict[str, Path]:
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    training = tmp_path / "training.json"
    training.write_text(
        json.dumps(
            {
                "environment_steps": 50_000,
                "style": "defensive",
                "test_cases_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    episodes = smoke / "episodes.jsonl"
    episodes.write_text("\n".join("{}" for _ in range(20)) + "\n", encoding="utf-8")
    summary = smoke / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "checkpoint_sha256": {"defensive": sha256_file(candidate)},
                "complete": True,
                "gates": {
                    "protocol_clean": True,
                    "skill_retention": retention,
                },
                "passed": False,
                "protective_presence_delta": primary,
                "protocol_inconsistencies": 0,
                "test_cases_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    (smoke / "manifest.json").write_text(
        json.dumps(
            {
                "episodes": 20,
                "episodes_sha256": sha256_file(episodes),
                "summary_sha256": sha256_file(summary),
            }
        ),
        encoding="utf-8",
    )
    return {"candidate": candidate, "smoke": smoke, "training": training}


@pytest.mark.parametrize(
    ("retention", "primary", "expected"),
    [
        (True, 0.1, "continue_to_100k"),
        (True, -0.1, "stop_50k"),
        (False, 0.1, "stop_50k"),
    ],
)
def test_m5_v2_frozen_continuation_rule(
    tmp_path: Path, retention: bool, primary: float, expected: str
) -> None:
    paths = _evidence(tmp_path, retention=retention, primary=primary)

    result = decide_m5_v2_candidate(
        style="defensive",
        training_summary=paths["training"],
        candidate=paths["candidate"],
        smoke_dir=paths["smoke"],
    )

    assert result["disposition"] == expected
