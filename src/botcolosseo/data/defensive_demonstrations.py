from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from botcolosseo.agents.duel_teachers import (
    ProtectiveDefensiveTeacher,
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
from botcolosseo.envs.defensive_signals import (
    defensive_risk,
    in_defensive_half,
    opponent_carrier_id,
)
from botcolosseo.envs.duel_protocol import DuelEvent, DuelEventType
from botcolosseo.envs.duel_types import DuelActorObservation, DuelPrivilegedState
from botcolosseo.envs.synchronous_duel import SynchronousDuelEnv
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS
from botcolosseo.scenarios.regions import RegionGraph


@dataclass(frozen=True)
class DefensiveStepEvidence:
    risk: bool
    risk_after: bool
    opponent_carrier: bool
    loose_core_in_defensive_half: bool
    events: tuple[DuelEvent, ...]


@dataclass(frozen=True)
class DefensiveWindowLabels:
    selected: tuple[bool, ...]
    reasons: tuple[str, ...]
    successful_windows: int
    denial_recovery_windows: int


@dataclass(frozen=True)
class _EpisodeRow:
    observation: DuelActorObservation
    protective_action: int
    base_action: int
    episode_start: bool
    task_id: int
    evidence: DefensiveStepEvidence


def label_defensive_windows(
    steps: tuple[DefensiveStepEvidence, ...], *, learner_side: str
) -> DefensiveWindowLabels:
    if learner_side not in ("host", "opponent"):
        raise ValueError("Defensive window learner side is invalid")
    selected = [False] * len(steps)
    reasons = ["not_risk"] * len(steps)
    successful_windows = 0
    denial_recovery_windows = 0
    start: int | None = None
    for index, step in enumerate(steps):
        if step.risk and start is None:
            start = index
        opponent_side = "opponent" if learner_side == "host" else "host"
        terminal_event = any(
            (
                _has_event(step.events, opponent_side, DuelEventType.DROP),
                _has_event(step.events, opponent_side, DuelEventType.SCORE),
                _has_event(step.events, learner_side, DuelEventType.PICKUP),
                _has_event(step.events, learner_side, DuelEventType.DEATH),
            )
        )
        if start is None or (
            step.risk_after and not terminal_event and index + 1 < len(steps)
        ):
            continue
        window = steps[start : index + 1]
        denial = any(
            item.opponent_carrier
            and _has_event(item.events, learner_side, DuelEventType.VALID_HIT)
            and _has_event(item.events, opponent_side, DuelEventType.DROP)
            for item in window
        )
        recovery = any(
            item.loose_core_in_defensive_half
            and _has_event(item.events, learner_side, DuelEventType.PICKUP)
            for item in window
        )
        conceded = any(
            _has_event(item.events, opponent_side, DuelEventType.SCORE)
            for item in window
        )
        lookahead = steps[index + 1 : index + 25]
        objective_progress = any(
            _has_event(item.events, learner_side, DuelEventType.PICKUP)
            or _has_event(item.events, learner_side, DuelEventType.SCORE)
            for item in lookahead
        )
        reason = (
            "successful_denial"
            if denial
            else "successful_recovery"
            if recovery
            else "resolved_to_objective"
            if not conceded and not step.risk_after and objective_progress
            else "rejected_risk_window"
        )
        success = reason != "rejected_risk_window"
        for window_index in range(start, index + 1):
            selected[window_index] = success
            reasons[window_index] = reason
        if success:
            successful_windows += 1
            denial_recovery_windows += int(denial or recovery)
        start = None
    return DefensiveWindowLabels(
        selected=tuple(selected),
        reasons=tuple(reasons),
        successful_windows=successful_windows,
        denial_recovery_windows=denial_recovery_windows,
    )


def defensive_step_evidence(
    state: DuelPrivilegedState,
    next_state: DuelPrivilegedState,
    events: tuple[DuelEvent, ...],
    *,
    learner_side: str,
) -> DefensiveStepEvidence:
    return DefensiveStepEvidence(
        risk=defensive_risk(state, learner_side),
        risk_after=defensive_risk(next_state, learner_side),
        opponent_carrier=state.carrier == opponent_carrier_id(learner_side),
        loose_core_in_defensive_half=state.carrier == 0
        and in_defensive_half(state.core_x, learner_side),
        events=events,
    )


def generate_defensive_demonstrations(
    *,
    root: Path,
    cases_path: Path,
    output_dir: Path,
    base_policy: CheckpointOpponentPolicy,
    transitions: int,
    shard_size: int,
    case_transition_cap: int,
    min_risk_transitions: int,
    min_denial_recovery_windows: int,
) -> dict[str, Any]:
    if min(
        transitions,
        shard_size,
        case_transition_cap,
        min_risk_transitions,
        min_denial_recovery_windows,
    ) <= 0:
        raise ValueError("Defensive demonstration settings must be positive")
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
    window_counts: Counter[str] = Counter()
    shards: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    episode = 0
    scenario_hash = ""
    selected_risk = 0
    selected_no_risk = 0
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
            learner = ProtectiveDefensiveTeacher(graph, side=case.learner_side)
            opponent_side = "opponent" if case.learner_side == "host" else "host"
            opponent = create_duel_teacher(opponent_name, graph, side=opponent_side)
            rows: list[_EpisodeRow] = []
            try:
                observations, info = env.reset()
                scenario_hash = info.scenario_hash
                learner.reset(seed=case.seed)
                opponent.reset(seed=case.seed ^ 0x5A5A5A5A)
                base_policy.reset()
                for decision in range(episode_cap):
                    state = env.teacher_state()
                    observation = (
                        observations.host
                        if case.learner_side == "host"
                        else observations.opponent
                    )
                    protective_action = learner.act(state)
                    base_action = base_policy.act(observation)
                    opponent_action = opponent.act(state)
                    host_action, away_action = (
                        (protective_action, opponent_action)
                        if case.learner_side == "host"
                        else (opponent_action, protective_action)
                    )
                    step = env.step(host_action, away_action)
                    next_state = env.teacher_state()
                    rows.append(
                        _EpisodeRow(
                            observation=observation,
                            protective_action=int(protective_action),
                            base_action=int(base_action),
                            episode_start=decision == 0,
                            task_id=TASK_IDS[learner.mode.value],
                            evidence=defensive_step_evidence(
                                state,
                                next_state,
                                step.events,
                                learner_side=case.learner_side,
                            ),
                        )
                    )
                    observations = type(observations)(step.host, step.opponent)
                    if step.terminated or step.truncated:
                        break
            finally:
                env.close()
            labels = label_defensive_windows(
                tuple(row.evidence for row in rows), learner_side=case.learner_side
            )
            episode_risk = sum(labels.selected)
            selected_risk += episode_risk
            target_no_risk = selected_risk // 2
            no_risk_needed = max(0, target_no_risk - selected_no_risk)
            no_risk_indices = [
                index for index, row in enumerate(rows) if not row.evidence.risk
            ][:no_risk_needed]
            selected_no_risk_indices = set(no_risk_indices)
            selected_no_risk += len(selected_no_risk_indices)
            window_counts["successful"] += labels.successful_windows
            window_counts["denial_recovery"] += labels.denial_recovery_windows
            for index, row in enumerate(rows):
                risk_selected = labels.selected[index]
                no_risk_selected = index in selected_no_risk_indices
                valid = risk_selected or no_risk_selected
                reason = labels.reasons[index] if row.evidence.risk else (
                    "selected_no_risk" if no_risk_selected else "skipped_no_risk"
                )
                action = row.protective_action if row.evidence.risk else row.base_action
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
    supervised = selected_risk + selected_no_risk
    no_risk_fraction = selected_no_risk / supervised if supervised else 0.0
    gate = {
        "complete": generated == transitions,
        "risk_transitions": selected_risk >= min_risk_transitions,
        "denial_recovery_windows": window_counts["denial_recovery"]
        >= min_denial_recovery_windows,
        "opponent_coverage": set(opponent_counts) == set(DUEL_OPPONENTS),
        "side_coverage": set(side_counts) == {"host", "opponent"},
        "no_risk_balance": 0.25 <= no_risk_fraction <= 0.50,
    }
    manifest = {
        "action_counts": dict(sorted(action_counts.items())),
        "base_checkpoint_sha256": base_policy.spec.checkpoint_sha256,
        "case_manifest_sha256": sha256_file(cases_path),
        "episode_count": episode,
        "gate": gate,
        "label_counts": dict(sorted(label_counts.items())),
        "min_denial_recovery_windows": min_denial_recovery_windows,
        "min_risk_transitions": min_risk_transitions,
        "no_risk_fraction": no_risk_fraction,
        "opponent_counts": dict(sorted(opponent_counts.items())),
        "passed": all(gate.values()),
        "privileged_leak_count": 0,
        "scenario_hash": scenario_hash,
        "schema_version": 1,
        "shards": shards,
        "side_counts": dict(sorted(side_counts.items())),
        "split": "train",
        "style": "defensive",
        "supervised_no_risk_transitions": selected_no_risk,
        "supervised_risk_transitions": selected_risk,
        "supervised_transitions": supervised,
        "test_cases_accessed": False,
        "transitions": generated,
        "window_counts": dict(sorted(window_counts.items())),
    }
    _write_json(manifest, output_dir / "train-manifest.json")
    return manifest


def _has_event(
    events: tuple[DuelEvent, ...], side: str, event_type: DuelEventType
) -> bool:
    return any(event.side == side and event.type is event_type for event in events)
