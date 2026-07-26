from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from botcolosseo.data.extraction_demonstrations import ExtractionCase

SCRIPT_STYLES = ("strong", "aggressive", "defensive", "explorer")


@dataclass(frozen=True)
class ExtractionEvaluationSplit:
    name: str
    seed_start: int
    pairs_per_opponent: int
    layout_id: str

    @property
    def episode_count(self) -> int:
        return self.pairs_per_opponent * len(SCRIPT_STYLES) * 2

    def cases(self) -> tuple[ExtractionCase, ...]:
        cases: list[ExtractionCase] = []
        seed = self.seed_start
        for opponent_style in SCRIPT_STYLES:
            for _ in range(self.pairs_per_opponent):
                cases.extend(
                    (
                        ExtractionCase(
                            self.name,
                            seed,
                            "host",
                            opponent_style,
                            self.layout_id,
                        ),
                        ExtractionCase(
                            self.name,
                            seed,
                            "opponent",
                            opponent_style,
                            self.layout_id,
                        ),
                    )
                )
                seed += 1
        return tuple(cases)


@dataclass(frozen=True)
class ExtractionEvaluationProtocol:
    schema_version: int
    splits: dict[str, ExtractionEvaluationSplit]
    sha256: str

    def cases(self, split: str) -> tuple[ExtractionCase, ...]:
        if split not in self.splits:
            raise ValueError("Unknown Extraction evaluation split")
        return self.splits[split].cases()


def load_extraction_evaluation_protocol(
    path: Path,
) -> ExtractionEvaluationProtocol:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported Extraction evaluation protocol")
    raw_splits = payload.get("splits")
    if set(raw_splits) != {"validation", "heldout", "test"}:
        raise ValueError("Extraction evaluation splits are incomplete")
    splits = {
        name: ExtractionEvaluationSplit(
            name=name,
            seed_start=int(item["seed_start"]),
            pairs_per_opponent=int(item["pairs_per_opponent"]),
            layout_id=str(item["layout_id"]),
        )
        for name, item in raw_splits.items()
    }
    if (
        splits["validation"].episode_count != 240
        or splits["heldout"].episode_count != 120
        or splits["test"].episode_count != 400
    ):
        raise ValueError("Extraction evaluation episode budgets drifted")
    cases = [case for split in splits.values() for case in split.cases()]
    identities = {(case.split, case.seed, case.learner_side) for case in cases}
    if len(identities) != len(cases):
        raise ValueError("Extraction evaluation cases overlap")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ExtractionEvaluationProtocol(1, splits, digest)


def protocol_manifest(protocol: ExtractionEvaluationProtocol) -> dict[str, object]:
    return {
        "schema_version": protocol.schema_version,
        "protocol_sha256": protocol.sha256,
        "splits": {
            name: {
                "episode_count": split.episode_count,
                "layout_id": split.layout_id,
                "seed_start": split.seed_start,
                "pairs_per_opponent": split.pairs_per_opponent,
            }
            for name, split in protocol.splits.items()
        },
        "test_cases_accessed": False,
    }
