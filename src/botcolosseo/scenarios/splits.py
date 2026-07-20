from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import yaml


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


@dataclass(frozen=True)
class TaskVariant:
    task: TaskKind
    map_name: str
    timeout_tics: int


def load_task_variants(path: Path) -> dict[TaskKind, TaskVariant]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != 1:
        raise ValueError("Task variants require protocol_version 1")
    variants: dict[TaskKind, TaskVariant] = {}
    for raw_task, item in payload.get("variants", {}).items():
        task = TaskKind(raw_task)
        map_name = str(item["map"]).upper()
        timeout_tics = int(item["timeout_tics"])
        if re.fullmatch(r"MAP[0-9]{2}", map_name) is None:
            raise ValueError(f"Invalid Doom map marker: {map_name}")
        if timeout_tics <= 0:
            raise ValueError(f"timeout_tics must be positive for {task.value}")
        variants[task] = TaskVariant(task, map_name, timeout_tics)
    missing = set(TaskKind).difference(variants)
    if missing:
        raise ValueError(f"Missing task variants: {sorted(task.value for task in missing)}")
    return variants


def generate_split_cases(
    *,
    master_seed: int,
    cases_per_task: int,
) -> dict[str, tuple[EpisodeCase, ...]]:
    if cases_per_task <= 0:
        raise ValueError("cases_per_task must be positive")
    split_names = ("train", "validation", "test")
    tasks = tuple(TaskKind)
    seed_rng = np.random.default_rng(master_seed)
    used_seeds: set[int] = set()
    result: dict[str, tuple[EpisodeCase, ...]] = {}
    for split in split_names:
        cases: list[EpisodeCase] = []
        for task in tasks:
            for _ in range(cases_per_task):
                seed = int(seed_rng.integers(0, 2**31, dtype=np.int64))
                while seed in used_seeds:
                    seed = int(seed_rng.integers(0, 2**31, dtype=np.int64))
                used_seeds.add(seed)
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
