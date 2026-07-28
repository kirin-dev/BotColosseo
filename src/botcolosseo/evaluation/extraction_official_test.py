from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from botcolosseo.data.extraction_demonstrations import ExtractionCase
from botcolosseo.evaluation.extraction_protocol import SCRIPT_STYLES

OFFICIAL_TEST_EPISODES = 400


@dataclass(frozen=True)
class SealedExtractionOfficialTest:
    scenario_hash: str
    validation_protocol_sha256: str
    cases: tuple[ExtractionCase, ...]


def load_sealed_extraction_official_test(
    path: Path,
) -> SealedExtractionOfficialTest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("split") != "test"
        or payload.get("test_cases_executed") is not False
        or payload.get("episode_count") != OFFICIAL_TEST_EPISODES
    ):
        raise ValueError("Sealed Extraction official-test identity is invalid")
    scenario_hash = payload.get("scenario_hash")
    validation_hash = payload.get("validation_protocol_sha256")
    if (
        not isinstance(scenario_hash, str)
        or len(scenario_hash) != 64
        or not isinstance(validation_hash, str)
        or len(validation_hash) != 64
    ):
        raise ValueError("Sealed Extraction official-test hashes are invalid")
    cases = tuple(ExtractionCase(**item) for item in payload.get("cases", ()))
    if len(cases) != OFFICIAL_TEST_EPISODES:
        raise ValueError("Sealed Extraction official-test budget drifted")
    identities = {
        (case.seed, case.learner_side, case.opponent_style, case.layout_id)
        for case in cases
    }
    if len(identities) != len(cases):
        raise ValueError("Sealed Extraction official-test cases overlap")
    if any(case.split != "test" or case.layout_id != "heldout-a" for case in cases):
        raise ValueError("Sealed Extraction official-test case scope drifted")
    coverage = Counter((case.opponent_style, case.learner_side) for case in cases)
    expected = {
        (style, side): 50
        for style in SCRIPT_STYLES
        for side in ("host", "opponent")
    }
    if coverage != expected:
        raise ValueError("Sealed Extraction official-test coverage is unbalanced")
    return SealedExtractionOfficialTest(
        scenario_hash=scenario_hash,
        validation_protocol_sha256=validation_hash,
        cases=cases,
    )
