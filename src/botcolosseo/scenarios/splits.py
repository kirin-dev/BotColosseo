from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class TaskKind(str, Enum):
    NAVIGATION = "navigation"
    PICKUP = "pickup"
    RETURN = "return"
    STATIC_HIT = "static_hit"
    MOVING_HIT = "moving_hit"


@dataclass(frozen=True)
class EpisodeCase:
    split: str
    task: TaskKind
    seed: int
    spawn_index: int
    target_index: int
    route: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["task"] = self.task.value
        return payload


def generate_split_cases(
    *,
    master_seed: int,
    cases_per_task: int,
) -> dict[str, tuple[EpisodeCase, ...]]:
    if cases_per_task <= 0:
        raise ValueError("cases_per_task must be positive")
    split_names = ("train", "validation", "test")
    tasks = tuple(TaskKind)
    seed_root = master_seed * 10_000_000
    result: dict[str, tuple[EpisodeCase, ...]] = {}
    for split_index, split in enumerate(split_names):
        cases: list[EpisodeCase] = []
        for task_index, task in enumerate(tasks):
            for case_index in range(cases_per_task):
                seed = (
                    seed_root
                    + split_index * 1_000_000
                    + task_index * cases_per_task
                    + case_index
                )
                rng = np.random.default_rng(seed)
                cases.append(
                    EpisodeCase(
                        split=split,
                        task=task,
                        seed=seed,
                        spawn_index=int(rng.integers(0, 3)),
                        target_index=int(rng.integers(0, 3)),
                        route=("direct_upper", "direct_lower", "flank")[
                            int(rng.integers(0, 3))
                        ],
                    )
                )
        result[split] = tuple(cases)
    return result


def write_split_manifests(
    splits: dict[str, tuple[EpisodeCase, ...]],
    output_dir: Path,
) -> tuple[Path, ...]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for split in ("train", "validation", "test"):
        if split not in splits:
            raise ValueError(f"Missing split: {split}")
        path = output_dir / f"{split}.json"
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            payload = [case.to_dict() for case in splits[split]]
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        paths.append(path)
    return tuple(paths)
