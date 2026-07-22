import json
from collections import Counter
from pathlib import Path

from botcolosseo.scenarios.duel_splits import (
    DUEL_OPPONENTS,
    generate_duel_splits,
    write_duel_manifests,
)


def test_duel_splits_are_reproducible_disjoint_balanced_and_side_swapped() -> None:
    first = generate_duel_splits(master_seed=20260720, pairs_per_opponent=50)
    second = generate_duel_splits(master_seed=20260720, pairs_per_opponent=50)

    assert first == second
    assert all(len(cases) == 500 for cases in first.values())
    seed_sets = [{case.seed for case in first[split]} for split in first]
    assert seed_sets[0].isdisjoint(seed_sets[1])
    assert seed_sets[0].isdisjoint(seed_sets[2])
    assert seed_sets[1].isdisjoint(seed_sets[2])
    for cases in first.values():
        assert Counter(case.opponent for case in cases) == Counter(
            {opponent: 100 for opponent in DUEL_OPPONENTS}
        )
        pairs = Counter((case.opponent, case.seed) for case in cases)
        assert set(pairs.values()) == {2}
        assert all(0 <= case.seed <= 2**31 - 1 for case in cases)


def test_duel_manifest_serialization_is_stable(tmp_path: Path) -> None:
    splits = generate_duel_splits(master_seed=7, pairs_per_opponent=2)

    paths = write_duel_manifests(splits, tmp_path)

    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert [path.name for path in paths] == ["train.json", "validation.json", "test.json"]
    assert payload[0]["learner_side"] == "host"
    assert payload[1]["learner_side"] == "opponent"
    assert paths[0].read_text(encoding="utf-8").endswith("\n")
