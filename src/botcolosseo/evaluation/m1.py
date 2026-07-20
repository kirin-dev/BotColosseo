from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Protocol

from botcolosseo.scenarios.splits import EpisodeCase, TaskKind

M1_THRESHOLDS = {
    TaskKind.NAVIGATION: 0.95,
    TaskKind.PICKUP: 0.95,
    TaskKind.RETURN: 0.95,
    TaskKind.STATIC_HIT: 0.90,
    TaskKind.MOVING_HIT: 0.75,
}

M1_TEACHERS = {
    TaskKind.NAVIGATION: "fixed_route",
    TaskKind.PICKUP: "objective_first",
    TaskKind.RETURN: "evasive_return",
    TaskKind.STATIC_HIT: "aggressive_script",
    TaskKind.MOVING_HIT: "aggressive_script",
}


class EpisodeResult(Protocol):
    success: bool
    truncated: bool
    decisions: int
    total_reward: float
    event_types: tuple[str, ...]
    scenario_hash: str


EpisodeRunner = Callable[[EpisodeCase, str], EpisodeResult]


@dataclass(frozen=True)
class CapabilityResult:
    task: str
    teacher: str
    successes: int
    trials: int
    success_rate: float
    wilson_lower: float
    wilson_upper: float
    threshold: float
    passed: bool


@dataclass(frozen=True)
class EvaluationSummary:
    official: bool
    passed: bool
    episodes: int
    protocol_inconsistencies: int
    capabilities: dict[str, CapabilityResult]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["capabilities"] = {
            task: asdict(result) for task, result in self.capabilities.items()
        }
        return payload


def wilson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between zero and trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials**2))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def evaluate_cases(
    cases: Sequence[EpisodeCase],
    *,
    runner: EpisodeRunner,
    official: bool,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[EvaluationSummary, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    successes = {task: 0 for task in TaskKind}
    trials = {task: 0 for task in TaskKind}
    inconsistencies = 0
    for index, case in enumerate(cases, start=1):
        teacher = M1_TEACHERS[case.task]
        result = runner(case, teacher)
        has_success_event = "task_success" in result.event_types
        inconsistent = bool(result.success) != has_success_event
        inconsistencies += int(inconsistent)
        trials[case.task] += 1
        successes[case.task] += int(result.success)
        rows.append(
            {
                "split": case.split,
                "task": case.task.value,
                "seed": case.seed,
                "spawn_index": case.spawn_index,
                "target_index": case.target_index,
                "route": case.route,
                "teacher": teacher,
                "success": bool(result.success),
                "truncated": bool(result.truncated),
                "decisions": result.decisions,
                "total_reward": result.total_reward,
                "event_types": "|".join(result.event_types),
                "protocol_inconsistent": inconsistent,
                "scenario_hash": result.scenario_hash,
            }
        )
        if progress is not None:
            progress(index, len(cases))

    capabilities: dict[str, CapabilityResult] = {}
    for task in TaskKind:
        task_trials = trials[task]
        if task_trials == 0:
            continue
        task_successes = successes[task]
        rate = task_successes / task_trials
        lower, upper = wilson_interval(task_successes, task_trials)
        threshold = M1_THRESHOLDS[task]
        capabilities[task.value] = CapabilityResult(
            task=task.value,
            teacher=M1_TEACHERS[task],
            successes=task_successes,
            trials=task_trials,
            success_rate=rate,
            wilson_lower=lower,
            wilson_upper=upper,
            threshold=threshold,
            passed=rate >= threshold,
        )
    complete = set(capabilities) == {task.value for task in TaskKind}
    gate_passed = (
        official
        and complete
        and inconsistencies == 0
        and all(result.passed for result in capabilities.values())
    )
    return (
        EvaluationSummary(
            official=official,
            passed=gate_passed,
            episodes=len(rows),
            protocol_inconsistencies=inconsistencies,
            capabilities=capabilities,
        ),
        rows,
    )


def load_cases(path: Path, expected_split: str) -> tuple[EpisodeCase, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = tuple(
        EpisodeCase(
            split=str(item["split"]),
            task=TaskKind(item["task"]),
            seed=int(item["seed"]),
            spawn_index=int(item["spawn_index"]),
            target_index=int(item["target_index"]),
            route=str(item["route"]),
        )
        for item in payload
    )
    if any(case.split != expected_split for case in cases):
        raise ValueError(f"Manifest contains rows outside split {expected_split}")
    seeds = [case.seed for case in cases]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds in {path}")
    return cases


def write_evidence(
    output_dir: Path,
    summary: EvaluationSummary,
    rows: Sequence[dict[str, object]],
    manifest: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "episodes.csv"
    csv_temp = output_dir / ".episodes.csv.tmp"
    fields = list(rows[0]) if rows else []
    with csv_temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    csv_temp.replace(csv_path)
    for name, payload in (("summary.json", summary.to_dict()), ("manifest.json", manifest)):
        path = output_dir / name
        temporary = output_dir / f".{name}.tmp"
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
