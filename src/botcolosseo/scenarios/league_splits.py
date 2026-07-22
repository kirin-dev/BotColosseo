from __future__ import annotations

import json
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from botcolosseo.scenarios.duel_splits import DuelCase

PAIR_COUNTS = {"train": 250, "validation": 50, "test": 50, "heldout": 50}
ROUTES = ("direct_upper", "direct_lower", "flank")
LEARNER_SIDES = ("host", "opponent")
_FIELDS = {
    "split",
    "pair_index",
    "seed",
    "learner_side",
    "core_spawn_index",
    "route",
}


@dataclass(frozen=True)
class LeagueCase:
    split: str
    pair_index: int
    seed: int
    learner_side: str
    core_spawn_index: int
    route: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_duel_case(self, opponent_id: str) -> DuelCase:
        if not opponent_id:
            raise ValueError("opponent_id must be non-empty")
        return DuelCase(opponent=opponent_id, **self.to_dict())


def generate_league_splits(
    *, master_seed: int = 20260721
) -> dict[str, tuple[LeagueCase, ...]]:
    rng = random.Random(master_seed)
    used_seeds: set[int] = set()
    result: dict[str, tuple[LeagueCase, ...]] = {}
    pair_index = 0
    for split, pair_count in PAIR_COUNTS.items():
        core_order = [0, 1, 2]
        route_order = list(ROUTES)
        rng.shuffle(core_order)
        rng.shuffle(route_order)
        cases: list[LeagueCase] = []
        for local_index in range(pair_count):
            seed = rng.randrange(2**31)
            while seed in used_seeds:
                seed = rng.randrange(2**31)
            used_seeds.add(seed)
            core_spawn_index = core_order[local_index % len(core_order)]
            route_index = local_index // len(core_order) + local_index % len(core_order)
            route = route_order[route_index % len(route_order)]
            for learner_side in LEARNER_SIDES:
                cases.append(
                    LeagueCase(
                        split=split,
                        pair_index=pair_index,
                        seed=seed,
                        learner_side=learner_side,
                        core_spawn_index=core_spawn_index,
                        route=route,
                    )
                )
            pair_index += 1
        result[split] = tuple(cases)
    return result


def write_league_manifests(
    cases: Mapping[str, Sequence[LeagueCase]], root: Path
) -> tuple[Path, ...]:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for split in PAIR_COUNTS:
        if split not in cases:
            raise ValueError(f"Missing league split: {split}")
        payload = json.dumps(
            [case.to_dict() for case in cases[split]], indent=2, sort_keys=True
        ) + "\n"
        path = root / f"{split}.json"
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=root,
                prefix=f".{split}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
            raise
        paths.append(path)
    return tuple(paths)


def load_league_cases(
    path: Path, *, expected_split: str, expected_pairs: int
) -> tuple[LeagueCase, ...]:
    if expected_split not in PAIR_COUNTS:
        raise ValueError(f"Unknown expected split: {expected_split}")
    if expected_pairs <= 0:
        raise ValueError("expected_pairs must be positive")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("League manifest must contain a list of rows")
    if len(payload) != expected_pairs * 2:
        raise ValueError(
            f"League manifest rows mismatch: expected {expected_pairs * 2}, got {len(payload)}"
        )

    parsed: list[LeagueCase] = []
    for row_index, row in enumerate(payload):
        if not isinstance(row, dict) or set(row) != _FIELDS:
            raise ValueError(f"League manifest row {row_index} has invalid fields")
        _validate_row(row, row_index=row_index, expected_split=expected_split)
        parsed.append(LeagueCase(**row))

    by_pair: dict[int, list[LeagueCase]] = defaultdict(list)
    for case in parsed:
        by_pair[case.pair_index].append(case)
    if len(by_pair) != expected_pairs:
        raise ValueError("League manifest pair count mismatch")
    seeds: set[int] = set()
    for pair_cases in by_pair.values():
        if [case.learner_side for case in pair_cases] != list(LEARNER_SIDES):
            raise ValueError("Each league pair must contain ordered host/opponent rows")
        if len({case.seed for case in pair_cases}) != 1:
            raise ValueError("League pair seed mismatch")
        if len({case.core_spawn_index for case in pair_cases}) != 1:
            raise ValueError("League pair core_spawn_index mismatch")
        if len({case.route for case in pair_cases}) != 1:
            raise ValueError("League pair route mismatch")
        seed = pair_cases[0].seed
        if seed in seeds:
            raise ValueError("League manifest reuses a seed across pairs")
        seeds.add(seed)
    return tuple(parsed)


def _validate_row(
    row: dict[str, object], *, row_index: int, expected_split: str
) -> None:
    if row["split"] != expected_split:
        raise ValueError(f"League manifest row {row_index} has invalid split")
    if type(row["pair_index"]) is not int or int(row["pair_index"]) < 0:
        raise ValueError(f"League manifest row {row_index} has invalid pair_index")
    if type(row["seed"]) is not int or not 0 <= int(row["seed"]) < 2**31:
        raise ValueError(f"League manifest row {row_index} has invalid seed")
    if row["learner_side"] not in LEARNER_SIDES:
        raise ValueError(f"League manifest row {row_index} has invalid learner_side")
    if type(row["core_spawn_index"]) is not int or row["core_spawn_index"] not in (0, 1, 2):
        raise ValueError(f"League manifest row {row_index} has invalid core_spawn_index")
    if row["route"] not in ROUTES:
        raise ValueError(f"League manifest row {row_index} has invalid route")
