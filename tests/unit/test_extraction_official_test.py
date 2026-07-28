from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from botcolosseo.cli.run_extraction_official_test import _load_partial
from botcolosseo.data.extraction_demonstrations import ExtractionCase
from botcolosseo.evaluation.extraction import ExtractionEpisodeMetrics
from botcolosseo.evaluation.extraction_official_test import (
    load_sealed_extraction_official_test,
)


def episode(case: ExtractionCase) -> ExtractionEpisodeMetrics:
    return ExtractionEpisodeMetrics(
        seed=case.seed,
        learner_side=case.learner_side,
        opponent_style=case.opponent_style,
        decisions=1,
        extracted_value=0,
        extracted=False,
        died=False,
        valid_hits=0,
        kills=0,
        cache_looted=0,
        attack_decisions=0,
        unique_route_cells=1,
        terminated=True,
        truncated=False,
    )


def test_official_test_partial_resume_checks_case_prefix(tmp_path: Path) -> None:
    cases = (
        ExtractionCase("test", 1, "host", "strong", "heldout-a"),
        ExtractionCase("test", 1, "opponent", "strong", "heldout-a"),
    )
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(asdict(episode(cases[0]))) + "\n", encoding="utf-8")

    assert _load_partial(path, cases) == [episode(cases[0])]

    path.write_text(json.dumps(asdict(episode(cases[1]))) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity drifted"):
        _load_partial(path, cases)


def test_sealed_official_test_requires_balanced_unseen_case_manifest(
    tmp_path: Path,
) -> None:
    cases = [
        ExtractionCase("test", 1000 + pair, side, style, "heldout-a")
        for style_index, style in enumerate(
            ("strong", "aggressive", "defensive", "explorer")
        )
        for pair in range(style_index * 50, (style_index + 1) * 50)
        for side in ("host", "opponent")
    ]
    path = tmp_path / "official-test-manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split": "test",
                "episode_count": 400,
                "scenario_hash": "a" * 64,
                "validation_protocol_sha256": "b" * 64,
                "cases": [asdict(case) for case in cases],
                "test_cases_executed": False,
            }
        ),
        encoding="utf-8",
    )

    sealed = load_sealed_extraction_official_test(path)

    assert len(sealed.cases) == 400
    assert sealed.scenario_hash == "a" * 64
