from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from botcolosseo.agents.duel_teachers import AggressiveDuelTeacher, create_duel_teacher
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
from botcolosseo.envs.synchronous_duel import SynchronousDuelEnv
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS
from botcolosseo.scenarios.regions import RegionGraph

ATTACK_ACTIONS = frozenset(
    {
        MacroAction.ATTACK,
        MacroAction.FORWARD_ATTACK,
        MacroAction.TURN_LEFT_ATTACK,
        MacroAction.TURN_RIGHT_ATTACK,
    }
)


def aggressive_supervision(
    action: MacroAction,
    events: tuple[DuelEvent, ...],
    *,
    learner_side: str,
    non_attack_index: int,
    non_attack_stride: int,
) -> tuple[bool, str]:
    if non_attack_stride <= 0 or non_attack_index < 0:
        raise ValueError("Invalid non-attack sampling settings")
    if action in ATTACK_ACTIONS:
        valid_hit = any(
            event.side == learner_side and event.type is DuelEventType.VALID_HIT for event in events
        )
        return valid_hit, "successful_attack" if valid_hit else "rejected_attack"
    selected = non_attack_index % non_attack_stride == 0
    return selected, "selected_non_attack" if selected else "skipped_non_attack"


def generate_aggressive_demonstrations(
    *,
    root: Path,
    cases_path: Path,
    output_dir: Path,
    transitions: int,
    shard_size: int,
    case_transition_cap: int,
    non_attack_stride: int,
) -> dict[str, Any]:
    if min(transitions, shard_size, case_transition_cap, non_attack_stride) <= 0:
        raise ValueError("Aggressive demonstration settings must be positive")
    cases = load_generation_cases(cases_path, expected_split="train")
    graph = RegionGraph.from_yaml(root / "assets/scenarios/crystal_run/src/regions.yaml")
    pools = {
        opponent: tuple(case for case in cases if case.opponent == opponent)
        for opponent in DUEL_OPPONENTS
    }
    cursors = {opponent: 0 for opponent in DUEL_OPPONENTS}
    target_by_opponent = {
        opponent: transitions // len(DUEL_OPPONENTS) + (index < transitions % len(DUEL_OPPONENTS))
        for index, opponent in enumerate(DUEL_OPPONENTS)
    }
    remaining = dict(target_by_opponent)
    action_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    opponent_counts: Counter[str] = Counter()
    shards: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    episode = 0
    non_attack_index = 0
    scenario_hash = ""
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
            learner = AggressiveDuelTeacher(graph, side=case.learner_side)
            opponent_side = "opponent" if case.learner_side == "host" else "host"
            opponent = create_duel_teacher(opponent_name, graph, side=opponent_side)
            try:
                observations, info = env.reset()
                scenario_hash = info.scenario_hash
                learner.reset(seed=case.seed)
                opponent.reset(seed=case.seed)
                episode_start = True
                for _ in range(episode_cap):
                    state = env.teacher_state()
                    learner_action = learner.act(state)
                    opponent_action = opponent.act(state)
                    learner_observation = (
                        observations.host if case.learner_side == "host" else observations.opponent
                    )
                    host_action, away_action = (
                        (learner_action, opponent_action)
                        if case.learner_side == "host"
                        else (opponent_action, learner_action)
                    )
                    step = env.step(host_action, away_action)
                    supervised, label = aggressive_supervision(
                        learner_action,
                        step.events,
                        learner_side=case.learner_side,
                        non_attack_index=non_attack_index,
                        non_attack_stride=non_attack_stride,
                    )
                    if learner_action not in ATTACK_ACTIONS:
                        non_attack_index += 1
                    buffer.append(
                        learner_observation,
                        teacher_action=int(learner_action),
                        episode_start=episode_start,
                        opponent_id=OPPONENT_IDS[opponent_name],
                        task_id=TASK_IDS[learner.mode.value],
                        train_seed=case.seed,
                        valid=supervised,
                    )
                    action_counts[learner_action.name] += 1
                    label_counts[label] += 1
                    opponent_counts[opponent_name] += 1
                    remaining[opponent_name] -= 1
                    episode_start = False
                    observations = type(observations)(step.host, step.opponent)
                    if step.terminated or step.truncated:
                        break
            finally:
                env.close()
            episode += 1
        shard_path = output_dir / f"train-{len(shards):05d}.npz"
        arrays = buffer.arrays()
        write_demonstration_shard(arrays, shard_path, require_all_valid=False)
        shards.append(
            {
                "file": shard_path.name,
                "sha256": sha256_file(shard_path),
                "trajectory_sha256": trajectory_sha256(arrays, require_all_valid=False),
                "transitions": len(buffer),
            }
        )
        generated += len(buffer)
    if label_counts["successful_attack"] == 0:
        raise RuntimeError("Aggressive demonstrations contain no successful attacks")
    manifest = {
        "action_counts": dict(sorted(action_counts.items())),
        "case_manifest_sha256": sha256_file(cases_path),
        "episode_count": episode,
        "label_counts": dict(sorted(label_counts.items())),
        "non_attack_stride": non_attack_stride,
        "opponent_counts": dict(sorted(opponent_counts.items())),
        "privileged_leak_count": 0,
        "scenario_hash": scenario_hash,
        "schema_version": 1,
        "shards": shards,
        "split": "train",
        "style": "aggressive",
        "supervised_transitions": int(
            label_counts["successful_attack"] + label_counts["selected_non_attack"]
        ),
        "test_cases_accessed": False,
        "transitions": generated,
    }
    _write_json(manifest, output_dir / "train-manifest.json")
    return manifest
