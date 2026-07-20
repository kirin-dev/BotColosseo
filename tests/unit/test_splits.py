import json
from pathlib import Path

from botcolosseo.scenarios.splits import (
    TaskKind,
    generate_split_cases,
    write_split_manifests,
)


def test_split_cases_are_reproducible_and_disjoint() -> None:
    first = generate_split_cases(master_seed=20260720, cases_per_task=100)
    second = generate_split_cases(master_seed=20260720, cases_per_task=100)

    assert first == second
    assert set(first) == {"train", "validation", "test"}
    assert all(len(first[name]) == 500 for name in first)
    seed_sets = [{case.seed for case in first[name]} for name in first]
    assert seed_sets[0].isdisjoint(seed_sets[1])
    assert seed_sets[0].isdisjoint(seed_sets[2])
    assert seed_sets[1].isdisjoint(seed_sets[2])
    assert {case.task for case in first["test"]} == set(TaskKind)


def test_split_manifest_serialization_is_stable(tmp_path: Path) -> None:
    cases = generate_split_cases(master_seed=7, cases_per_task=2)

    paths = write_split_manifests(cases, tmp_path)

    assert [path.name for path in paths] == ["train.json", "validation.json", "test.json"]
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload[0]["split"] == "train"
    assert paths[0].read_text(encoding="utf-8").endswith("\n")
