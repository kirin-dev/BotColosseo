from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DUEL_OPPONENTS = (
    "random_legal",
    "fixed_route",
    "objective_first",
    "aggressive_script",
    "defensive_script",
)


@dataclass(frozen=True)
class DuelCase:
    split: str
    pair_index: int
    seed: int
    opponent: str
    learner_side: str
    core_spawn_index: int
    route: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def generate_duel_splits(
    *, master_seed: int, pairs_per_opponent: int
) -> dict[str, tuple[DuelCase, ...]]:
    if pairs_per_opponent <= 0:
        raise ValueError("pairs_per_opponent must be positive")
    seed_rng = np.random.default_rng(master_seed)
    used_seeds: set[int] = set()
    result: dict[str, tuple[DuelCase, ...]] = {}
    pair_index = 0
    for split in ("train", "validation", "test"):
        cases: list[DuelCase] = []
        for opponent in DUEL_OPPONENTS:
            for _ in range(pairs_per_opponent):
                seed = int(seed_rng.integers(0, 2**31, dtype=np.int64))
                while seed in used_seeds:
                    seed = int(seed_rng.integers(0, 2**31, dtype=np.int64))
                used_seeds.add(seed)
                case_rng = np.random.default_rng(seed)
                core_spawn_index = int(case_rng.integers(0, 3))
                route = ("direct_upper", "direct_lower", "flank")[
                    int(case_rng.integers(0, 3))
                ]
                for side in ("host", "opponent"):
                    cases.append(
                        DuelCase(
                            split=split,
                            pair_index=pair_index,
                            seed=seed,
                            opponent=opponent,
                            learner_side=side,
                            core_spawn_index=core_spawn_index,
                            route=route,
                        )
                    )
                pair_index += 1
        result[split] = tuple(cases)
    return result


def write_duel_manifests(
    splits: dict[str, tuple[DuelCase, ...]], output_dir: Path
) -> tuple[Path, ...]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for split in ("train", "validation", "test"):
        if split not in splits:
            raise ValueError(f"Missing duel split: {split}")
        path = output_dir / f"{split}.json"
        temporary = output_dir / f".{split}.json.tmp"
        try:
            temporary.write_text(
                json.dumps(
                    [case.to_dict() for case in splits[split]], indent=2, sort_keys=True
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        paths.append(path)
    return tuple(paths)
