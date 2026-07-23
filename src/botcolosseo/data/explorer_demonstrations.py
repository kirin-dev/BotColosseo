from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from botcolosseo.agents.duel_teachers import (
    EXPLORER_ROUTE_CYCLE,
    RouteExplorerTeacher,
    create_duel_teacher,
)
from botcolosseo.agents.league_opponents import CheckpointOpponentPolicy
from botcolosseo.data.demonstrations import (
    OPPONENT_IDS,
    TASK_IDS,
    DemonstrationBuffer,
    _write_json,
    load_generation_cases,
    sha256_file,
    trajectory_sha256,
    write_demonstration_shard,
)
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_protocol import DuelEvent, DuelEventType
from botcolosseo.envs.duel_types import DuelActorObservation
from botcolosseo.envs.synchronous_duel import SynchronousDuelEnv
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS
from botcolosseo.scenarios.regions import RegionGraph

_ROUTE_REGIONS = {
    "direct_upper": {"upper_route"},
    "direct_lower": {"lower_route"},
    "flank": {"flank_west", "flank_east"},
}
_FLANK_REGIONS = _ROUTE_REGIONS["flank"]


@dataclass(frozen=True)
class ExplorerStepEvidence:
    selected_route: str
    learner_region: str | None
    events: tuple[DuelEvent, ...]

    def __post_init__(self) -> None:
        if self.selected_route not in EXPLORER_ROUTE_CYCLE:
            raise ValueError("Explorer evidence route is invalid")


@dataclass(frozen=True)
class ExplorerWindowLabels:
    selected: tuple[bool, ...]
    reasons: tuple[str, ...]
    successful_windows: int
    route_windows: tuple[tuple[str, int], ...]
    windows: tuple[ExplorerWindow, ...]


@dataclass(frozen=True)
class ExplorerWindow:
    start: int
    stop: int
    route: str
    successful: bool


@dataclass(frozen=True)
class ExplorerWindowAdmission:
    selected: tuple[bool, ...]
    reasons: tuple[str, ...]
    accepted_windows: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _EpisodeRow:
    observation: DuelActorObservation
    route_action: int
    base_action: int
    episode_start: bool
    task_id: int
    evidence: ExplorerStepEvidence


def label_explorer_windows(
    steps: tuple[ExplorerStepEvidence, ...], *, learner_side: str
) -> ExplorerWindowLabels:
    if learner_side not in ("host", "opponent"):
        raise ValueError("Explorer window learner side is invalid")
    opponent_side = "opponent" if learner_side == "host" else "host"
    selected = [False] * len(steps)
    reasons = ["incomplete_route_window"] * len(steps)
    route_counts = {route: 0 for route in EXPLORER_ROUTE_CYCLE}
    windows: list[ExplorerWindow] = []
    start = 0
    for index, step in enumerate(steps):
        learner_score = _has_event(step.events, learner_side, DuelEventType.SCORE)
        opponent_score = _has_event(step.events, opponent_side, DuelEventType.SCORE)
        learner_death = _has_event(step.events, learner_side, DuelEventType.DEATH)
        terminal = learner_score or opponent_score or learner_death or index + 1 == len(steps)
        if not terminal:
            continue
        window = steps[start : index + 1]
        routes = {item.selected_route for item in window}
        expected_route = step.selected_route
        matches = (
            len(routes) == 1
            and expected_route in routes
            and _matches_route(
                expected_route,
                {item.learner_region for item in window if item.learner_region},
            )
        )
        success = learner_score and not opponent_score and not learner_death and matches
        reason = f"successful_{expected_route}" if success else "rejected_route_window"
        for window_index in range(start, index + 1):
            selected[window_index] = success
            reasons[window_index] = reason
        if success:
            route_counts[expected_route] += 1
        windows.append(ExplorerWindow(start, index + 1, expected_route, success))
        start = index + 1
    return ExplorerWindowLabels(
        selected=tuple(selected),
        reasons=tuple(reasons),
        successful_windows=sum(route_counts.values()),
        route_windows=tuple((route, route_counts[route]) for route in EXPLORER_ROUTE_CYCLE),
        windows=tuple(windows),
    )


def admit_balanced_explorer_windows(
    labels: ExplorerWindowLabels,
    *,
    retained_route_windows: dict[str, int],
    max_per_route: int,
) -> ExplorerWindowAdmission:
    if (
        set(retained_route_windows) != set(EXPLORER_ROUTE_CYCLE)
        or any(value < 0 for value in retained_route_windows.values())
        or max_per_route <= 0
    ):
        raise ValueError("Explorer route admission settings are invalid")
    admitted = [False] * len(labels.selected)
    reasons = list(labels.reasons)
    accepted = {route: 0 for route in EXPLORER_ROUTE_CYCLE}
    for window in labels.windows:
        if not window.successful:
            continue
        used = retained_route_windows[window.route] + accepted[window.route]
        if used >= max_per_route:
            for index in range(window.start, window.stop):
                reasons[index] = "skipped_route_balance"
            continue
        accepted[window.route] += 1
        for index in range(window.start, window.stop):
            admitted[index] = True
    return ExplorerWindowAdmission(
        selected=tuple(admitted),
        reasons=tuple(reasons),
        accepted_windows=tuple(
            (route, accepted[route]) for route in EXPLORER_ROUTE_CYCLE
        ),
    )


def _matches_route(route: str, visited: set[str]) -> bool:
    if route == "flank":
        return _FLANK_REGIONS.issubset(visited)
    if not visited.isdisjoint(_FLANK_REGIONS):
        return False
    if route == "direct_upper":
        return "upper_route" in visited
    return "lower_route" in visited and "upper_route" not in visited


def _has_event(
    events: tuple[DuelEvent, ...], side: str, event_type: DuelEventType
) -> bool:
    return any(event.side == side and event.type is event_type for event in events)


def generate_explorer_demonstrations(
    *,
    root: Path,
    cases_path: Path,
    output_dir: Path,
    base_policy: CheckpointOpponentPolicy,
    transitions: int,
    shard_size: int,
    case_transition_cap: int,
    min_route_transitions: int,
    min_route_fraction: float,
    max_route_windows_per_route: int,
) -> dict[str, Any]:
    if min(
        transitions,
        shard_size,
        case_transition_cap,
        min_route_transitions,
        max_route_windows_per_route,
    ) <= 0 or not 0.0 < min_route_fraction <= 1.0 / len(EXPLORER_ROUTE_CYCLE):
        raise ValueError("Explorer demonstration settings are invalid")
    cases = load_generation_cases(cases_path, expected_split="train")
    graph = RegionGraph.from_yaml(root / "assets/scenarios/crystal_run/src/regions.yaml")
    pools = {
        opponent: tuple(case for case in cases if case.opponent == opponent)
        for opponent in DUEL_OPPONENTS
    }
    cursors = {opponent: 0 for opponent in DUEL_OPPONENTS}
    target_by_opponent = {
        opponent: transitions // len(DUEL_OPPONENTS)
        + (index < transitions % len(DUEL_OPPONENTS))
        for index, opponent in enumerate(DUEL_OPPONENTS)
    }
    remaining = dict(target_by_opponent)
    action_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    opponent_counts: Counter[str] = Counter()
    side_counts: Counter[str] = Counter()
    route_opponents: Counter[str] = Counter()
    route_sides: Counter[str] = Counter()
    route_windows: Counter[str] = Counter()
    raw_route_windows: Counter[str] = Counter()
    initial_scores: Counter[str] = Counter()
    shards: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    episode = 0
    scenario_hash = ""
    selected_route = 0
    selected_context = 0
    while generated < transitions:
        shard_target = min(shard_size, transitions - generated)
        buffer = DemonstrationBuffer(capacity=shard_target, allow_masked=True)
        while len(buffer) < shard_target:
            candidates = [name for name in DUEL_OPPONENTS if remaining[name] > 0]
            opponent_name = candidates[episode % len(candidates)]
            pool = pools[opponent_name]
            case = pool[cursors[opponent_name] % len(pool)]
            cursors[opponent_name] += 1
            episode_cap = min(
                case_transition_cap,
                remaining[opponent_name],
                shard_target - len(buffer),
            )
            env = SynchronousDuelEnv(
                config_path=root / "assets/scenarios/crystal_run/crystal_run.cfg",
                region_graph=graph,
                seed=case.seed,
                max_decisions=episode_cap,
            )
            learner = RouteExplorerTeacher(graph, side=case.learner_side)
            opponent_side = "opponent" if case.learner_side == "host" else "host"
            opponent = create_duel_teacher(opponent_name, graph, side=opponent_side)
            rows: list[_EpisodeRow] = []
            try:
                observations, info = env.reset()
                scenario_hash = info.scenario_hash
                learner.reset(seed=case.seed)
                opponent.reset(seed=case.seed ^ 0x5A5A5A5A)
                base_policy.reset()
                initial = (
                    observations.host.own_score
                    if case.learner_side == "host"
                    else observations.opponent.own_score
                )
                initial_scores[f"{case.learner_side}:{initial}"] += 1
                for decision in range(episode_cap):
                    state = env.teacher_state()
                    observation = (
                        observations.host
                        if case.learner_side == "host"
                        else observations.opponent
                    )
                    route_action = learner.act(state)
                    base_action = base_policy.act(observation)
                    opponent_action = opponent.act(state)
                    host_action, away_action = (
                        (route_action, opponent_action)
                        if case.learner_side == "host"
                        else (opponent_action, route_action)
                    )
                    step = env.step(host_action, away_action)
                    region = (
                        state.host_region
                        if case.learner_side == "host"
                        else state.opponent_region
                    )
                    rows.append(
                        _EpisodeRow(
                            observation=observation,
                            route_action=int(route_action),
                            base_action=int(base_action),
                            episode_start=decision == 0,
                            task_id=TASK_IDS[learner.mode.value],
                            evidence=ExplorerStepEvidence(
                                learner.route_name,
                                region,
                                step.events,
                            ),
                        )
                    )
                    observations = type(observations)(step.host, step.opponent)
                    if step.terminated or step.truncated:
                        break
            finally:
                env.close()
            labels = label_explorer_windows(
                tuple(row.evidence for row in rows),
                learner_side=case.learner_side,
            )
            raw_route_windows.update(dict(labels.route_windows))
            admission = admit_balanced_explorer_windows(
                labels,
                retained_route_windows={
                    route: route_windows[route] for route in EXPLORER_ROUTE_CYCLE
                },
                max_per_route=max_route_windows_per_route,
            )
            route_windows.update(dict(admission.accepted_windows))
            episode_route = sum(admission.selected)
            selected_route += episode_route
            target_context = selected_route // 2
            context_needed = max(0, target_context - selected_context)
            context_indices = [
                index
                for index, selected in enumerate(admission.selected)
                if not selected
            ][:context_needed]
            selected_context_indices = set(context_indices)
            selected_context += len(selected_context_indices)
            for index, row in enumerate(rows):
                route_selected = admission.selected[index]
                context_selected = index in selected_context_indices
                valid = route_selected or context_selected
                reason = (
                    admission.reasons[index]
                    if route_selected
                    else "selected_base_context"
                    if context_selected
                    else admission.reasons[index]
                )
                action = row.route_action if route_selected else row.base_action
                buffer.append(
                    row.observation,
                    teacher_action=action,
                    episode_start=row.episode_start,
                    opponent_id=OPPONENT_IDS[opponent_name],
                    task_id=row.task_id,
                    train_seed=case.seed,
                    valid=valid,
                )
                action_counts[MacroAction(action).name] += 1
                label_counts[reason] += 1
                opponent_counts[opponent_name] += 1
                side_counts[case.learner_side] += 1
                if route_selected:
                    route_opponents[opponent_name] += 1
                    route_sides[case.learner_side] += 1
                remaining[opponent_name] -= 1
            episode += 1
        shard_path = output_dir / f"train-{len(shards):05d}.npz"
        arrays = buffer.arrays()
        write_demonstration_shard(arrays, shard_path, require_all_valid=False)
        shards.append(
            {
                "file": shard_path.name,
                "sha256": sha256_file(shard_path),
                "trajectory_sha256": trajectory_sha256(
                    arrays, require_all_valid=False
                ),
                "transitions": len(buffer),
            }
        )
        generated += len(buffer)
    supervised = selected_route + selected_context
    context_fraction = selected_context / supervised if supervised else 0.0
    successful_windows = sum(route_windows.values())
    route_fractions = {
        route: route_windows[route] / successful_windows
        if successful_windows
        else 0.0
        for route in EXPLORER_ROUTE_CYCLE
    }
    gate = {
        "complete": generated == transitions,
        "route_transitions": selected_route >= min_route_transitions,
        "route_coverage": all(route_windows[route] > 0 for route in EXPLORER_ROUTE_CYCLE),
        "route_balance": all(
            route_fractions[route] + 1e-12 >= min_route_fraction
            for route in EXPLORER_ROUTE_CYCLE
        ),
        "opponent_coverage": set(route_opponents) == set(DUEL_OPPONENTS),
        "side_coverage": set(route_sides) == {"host", "opponent"},
        "context_balance": 0.25 <= context_fraction <= 0.50,
    }
    manifest = {
        "action_counts": dict(sorted(action_counts.items())),
        "base_checkpoint_sha256": base_policy.spec.checkpoint_sha256,
        "case_manifest_sha256": sha256_file(cases_path),
        "context_fraction": context_fraction,
        "episode_count": episode,
        "gate": gate,
        "initial_score_counts": dict(sorted(initial_scores.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "min_route_fraction": min_route_fraction,
        "min_route_transitions": min_route_transitions,
        "max_route_windows_per_route": max_route_windows_per_route,
        "opponent_counts": dict(sorted(opponent_counts.items())),
        "passed": all(gate.values()),
        "privileged_leak_count": 0,
        "route_fractions": route_fractions,
        "raw_route_window_counts": {
            route: raw_route_windows[route] for route in EXPLORER_ROUTE_CYCLE
        },
        "route_window_counts": {
            route: route_windows[route] for route in EXPLORER_ROUTE_CYCLE
        },
        "scenario_hash": scenario_hash,
        "schema_version": 1,
        "shards": shards,
        "side_counts": dict(sorted(side_counts.items())),
        "split": "train",
        "style": "explorer",
        "supervised_base_context_transitions": selected_context,
        "supervised_route_transitions": selected_route,
        "supervised_transitions": supervised,
        "test_cases_accessed": False,
        "transitions": generated,
    }
    _write_json(manifest, output_dir / "train-manifest.json")
    return manifest
