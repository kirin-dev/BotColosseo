from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtractionRunHistory:
    environment_steps: int
    episodes: int
    event_counts: Counter[str]
    reward_components: Counter[str]
    kl_early_stops: int


def reconcile_extraction_metrics(
    path: Path,
    *,
    committed_environment_steps: int,
) -> ExtractionRunHistory:
    if committed_environment_steps < 0:
        raise ValueError("Committed Extraction steps must be nonnegative")
    if not path.exists():
        if committed_environment_steps:
            raise FileNotFoundError("Extraction checkpoint exists without metrics")
        return ExtractionRunHistory(0, 0, Counter(), Counter(), 0)
    kept: list[dict[str, object]] = []
    group_steps = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            break
        if record.get("kind") == "rollout":
            group_steps = int(record["environment_steps"])
        record_steps = int(record.get("environment_steps", group_steps))
        if record_steps > committed_environment_steps:
            break
        kept.append(record)
    temporary = path.with_name(f".{path.name}.reconcile.tmp")
    temporary.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n" for record in kept
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
    events: Counter[str] = Counter()
    rewards: Counter[str] = Counter()
    episodes = 0
    kl_stops = 0
    last_steps = 0
    for record in kept:
        if record.get("kind") == "rollout":
            last_steps = int(record["environment_steps"])
            events.update(record["event_counts"])
            rewards.update(record["reward_components"])
        elif record.get("kind") == "episode":
            episodes += 1
        elif record.get("kind") == "kl_early_stop":
            kl_stops += 1
    if last_steps != committed_environment_steps:
        raise ValueError("Extraction metrics and checkpoint steps disagree")
    return ExtractionRunHistory(
        last_steps,
        episodes,
        events,
        rewards,
        kl_stops,
    )
