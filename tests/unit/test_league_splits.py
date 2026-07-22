import json
from collections import Counter
from pathlib import Path

import pytest

from botcolosseo.scenarios.league_splits import (
    PAIR_COUNTS,
    LeagueCase,
    generate_league_splits,
    load_league_cases,
    write_league_manifests,
)


def test_league_splits_are_reproducible_disjoint_and_side_swapped() -> None:
    first = generate_league_splits()
    second = generate_league_splits()

    assert first == second
    assert set(first) == set(PAIR_COUNTS)
    seed_sets: dict[str, set[int]] = {}
    for split, pair_count in PAIR_COUNTS.items():
        cases = first[split]
        assert len(cases) == pair_count * 2
        seed_sets[split] = {case.seed for case in cases}
        assert len(seed_sets[split]) == pair_count
        assert all(0 <= seed <= 2**31 - 1 for seed in seed_sets[split])

        by_pair: dict[int, list[LeagueCase]] = {}
        for case in cases:
            by_pair.setdefault(case.pair_index, []).append(case)
        assert len(by_pair) == pair_count
        for paired_cases in by_pair.values():
            assert [case.learner_side for case in paired_cases] == ["host", "opponent"]
            assert len({case.seed for case in paired_cases}) == 1
            assert len({case.core_spawn_index for case in paired_cases}) == 1
            assert len({case.route for case in paired_cases}) == 1

        core_counts = Counter(case.core_spawn_index for case in cases[::2])
        route_counts = Counter(case.route for case in cases[::2])
        assert set(core_counts) == {0, 1, 2}
        assert set(route_counts) == {"direct_upper", "direct_lower", "flank"}
        assert max(core_counts.values()) - min(core_counts.values()) <= 1
        assert max(route_counts.values()) - min(route_counts.values()) <= 1

    for left_index, left in enumerate(PAIR_COUNTS):
        for right in tuple(PAIR_COUNTS)[left_index + 1 :]:
            assert seed_sets[left].isdisjoint(seed_sets[right])


def test_league_case_adapts_to_neutral_duel_case() -> None:
    case = generate_league_splits(master_seed=7)["train"][0]

    duel_case = case.to_duel_case("history-0001")

    assert duel_case.opponent == "history-0001"
    assert duel_case.seed == case.seed
    assert duel_case.learner_side == case.learner_side


def test_league_manifest_round_trip_is_canonical(tmp_path: Path) -> None:
    splits = generate_league_splits(master_seed=11)

    paths = write_league_manifests(splits, tmp_path)

    assert [path.name for path in paths] == [
        "train.json",
        "validation.json",
        "test.json",
        "heldout.json",
    ]
    assert paths[0].read_text(encoding="utf-8").endswith("\n")
    assert load_league_cases(
        paths[0], expected_split="train", expected_pairs=PAIR_COUNTS["train"]
    ) == splits["train"]

    second_root = tmp_path / "second"
    second_paths = write_league_manifests(splits, second_root)
    assert [path.read_bytes() for path in paths] == [
        path.read_bytes() for path in second_paths
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update({"unexpected": True}), "fields"),
        (lambda row: row.update({"learner_side": "spectator"}), "learner_side"),
        (lambda row: row.update({"seed": 2**31}), "seed"),
        (lambda row: row.update({"split": "test"}), "split"),
    ],
)
def test_league_manifest_rejects_invalid_rows(
    tmp_path: Path, mutation: object, message: str
) -> None:
    splits = generate_league_splits(master_seed=13)
    path = write_league_manifests(splits, tmp_path)[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload[0])  # type: ignore[operator]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_league_cases(path, expected_split="train", expected_pairs=250)


def test_league_manifest_rejects_incomplete_pair(tmp_path: Path) -> None:
    splits = generate_league_splits(master_seed=17)
    path = write_league_manifests(splits, tmp_path)[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop()
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="rows"):
        load_league_cases(path, expected_split="train", expected_pairs=250)
