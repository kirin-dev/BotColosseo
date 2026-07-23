from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from botcolosseo.agents.duel_teachers import (
    DuelTeacherMode,
    ObjectiveDuelTeacher,
    create_duel_teacher,
)
from botcolosseo.data.schema import DEMONSTRATION_FIELDS, validate_demonstration_shard
from botcolosseo.envs.duel_types import DuelActorObservation
from botcolosseo.envs.synchronous_duel import SynchronousDuelEnv
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS, DuelCase
from botcolosseo.scenarios.regions import RegionGraph

OPPONENT_IDS = {name: index for index, name in enumerate(DUEL_OPPONENTS)}
TASK_IDS = {mode.value: index for index, mode in enumerate(DuelTeacherMode)}


class DemonstrationBuffer:
    def __init__(self, *, capacity: int, allow_masked: bool = False) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.allow_masked = allow_masked
        self._rows: list[tuple[object, ...]] = []

    def __len__(self) -> int:
        return len(self._rows)

    def append(
        self,
        observation: DuelActorObservation,
        *,
        teacher_action: int,
        episode_start: bool,
        opponent_id: int,
        task_id: int,
        train_seed: int,
        valid: bool = True,
    ) -> None:
        if len(self) >= self.capacity:
            raise BufferError("Demonstration buffer capacity exceeded")
        scalars = np.asarray(
            (
                observation.health / 200.0,
                observation.armor / 200.0,
                min(observation.ammo, 100.0) / 100.0,
                min(observation.own_score, 3) / 3.0,
                min(observation.opponent_score, 3) / 3.0,
                float(observation.has_core),
            ),
            dtype=np.float32,
        )
        self._rows.append(
            (
                np.array(observation.frame, copy=True),
                scalars,
                observation.previous_action,
                teacher_action,
                episode_start,
                valid,
                opponent_id,
                task_id,
                train_seed,
            )
        )

    def arrays(self) -> dict[str, np.ndarray]:
        if not self._rows:
            raise ValueError("Cannot materialize an empty demonstration buffer")
        columns = tuple(zip(*self._rows, strict=True))
        dtypes = (
            np.uint8,
            np.float32,
            np.int8,
            np.int8,
            np.bool_,
            np.bool_,
            np.uint8,
            np.uint8,
            np.int64,
        )
        arrays = {
            name: np.asarray(column, dtype=dtype)
            for name, column, dtype in zip(
                DEMONSTRATION_FIELDS, columns, dtypes, strict=True
            )
        }
        validate_demonstration_shard(arrays, require_all_valid=not self.allow_masked)
        return arrays


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def write_demonstration_shard(
    arrays: dict[str, np.ndarray],
    output_path: Path,
    *,
    require_all_valid: bool = True,
) -> Path:
    validate_demonstration_shard(arrays, require_all_valid=require_all_valid)
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in DEMONSTRATION_FIELDS:
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 0
                archive.writestr(info, _npy_bytes(arrays[name]), compresslevel=9)
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output_path


def load_demonstration_shard(
    path: Path, *, require_all_valid: bool = True
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    validate_demonstration_shard(arrays, require_all_valid=require_all_valid)
    return arrays


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trajectory_sha256(
    arrays: dict[str, np.ndarray], *, require_all_valid: bool = True
) -> str:
    validate_demonstration_shard(arrays, require_all_valid=require_all_valid)
    digest = hashlib.sha256()
    for name in DEMONSTRATION_FIELDS:
        if name == "frame":
            continue
        digest.update(name.encode("utf-8"))
        digest.update(_npy_bytes(arrays[name]))
    return digest.hexdigest()


def load_generation_cases(path: Path, *, expected_split: str) -> tuple[DuelCase, ...]:
    if expected_split == "test" or path.name == "test.json":
        raise ValueError("Demonstration generation may not access test cases")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = tuple(DuelCase(**item) for item in payload)
    if not cases or any(case.split != expected_split for case in cases):
        raise ValueError(f"Case manifest does not contain only {expected_split} cases")
    if {case.opponent for case in cases} != set(DUEL_OPPONENTS):
        raise ValueError("Generation cases must cover every frozen opponent")
    return cases


def _write_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output_path


def generate_demonstration_split(
    *,
    root: Path,
    split: str,
    cases_path: Path,
    output_dir: Path,
    transitions: int,
    shard_size: int,
    case_transition_cap: int,
) -> dict[str, Any]:
    if transitions <= 0 or shard_size <= 0 or case_transition_cap <= 0:
        raise ValueError("Generation sizes must be positive")
    cases = load_generation_cases(cases_path, expected_split=split)
    graph = RegionGraph.from_yaml(
        root / "assets/scenarios/crystal_run/src/regions.yaml"
    )
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
    task_counts: Counter[str] = Counter()
    opponent_counts: Counter[str] = Counter()
    shards: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    episode = 0
    scenario_hash = ""
    while generated < transitions:
        shard_target = min(shard_size, transitions - generated)
        buffer = DemonstrationBuffer(capacity=shard_target)
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
            learner = ObjectiveDuelTeacher(graph, side=case.learner_side)
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
                        observations.host
                        if case.learner_side == "host"
                        else observations.opponent
                    )
                    buffer.append(
                        learner_observation,
                        teacher_action=int(learner_action),
                        episode_start=episode_start,
                        opponent_id=OPPONENT_IDS[opponent_name],
                        task_id=TASK_IDS[learner.mode.value],
                        train_seed=case.seed,
                    )
                    action_counts[learner_action.name] += 1
                    task_counts[learner.mode.value] += 1
                    opponent_counts[opponent_name] += 1
                    remaining[opponent_name] -= 1
                    episode_start = False
                    host_action, away_action = (
                        (learner_action, opponent_action)
                        if case.learner_side == "host"
                        else (opponent_action, learner_action)
                    )
                    step = env.step(host_action, away_action)
                    observations = type(observations)(step.host, step.opponent)
                    if step.terminated or step.truncated:
                        break
            finally:
                env.close()
            episode += 1
        shard_index = len(shards)
        shard_path = output_dir / f"{split}-{shard_index:05d}.npz"
        arrays = buffer.arrays()
        write_demonstration_shard(arrays, shard_path)
        shards.append(
            {
                "file": shard_path.name,
                "sha256": sha256_file(shard_path),
                "trajectory_sha256": trajectory_sha256(arrays),
                "transitions": len(buffer),
            }
        )
        generated += len(buffer)
    manifest = {
        "action_counts": dict(sorted(action_counts.items())),
        "case_manifest_sha256": sha256_file(cases_path),
        "episode_count": episode,
        "frame_reproducibility": "engine-rendered-best-effort",
        "opponent_counts": dict(sorted(opponent_counts.items())),
        "privileged_leak_count": 0,
        "requested_transitions": transitions,
        "scenario_hash": scenario_hash,
        "schema": {
            "fields": list(DEMONSTRATION_FIELDS),
            "frame": {"dtype": "uint8", "shape": [84, 84]},
            "scalars": {"dtype": "float32", "shape": [6]},
        },
        "schema_version": 1,
        "shards": shards,
        "split": split,
        "task_counts": dict(sorted(task_counts.items())),
        "test_cases_accessed": False,
        "transitions": generated,
    }
    _write_json(manifest, output_dir / f"{split}-manifest.json")
    return manifest


def render_demonstration_distribution(
    manifests: list[dict[str, Any]], output_path: Path
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    opponents = list(DUEL_OPPONENTS)
    splits = [str(manifest["split"]) for manifest in manifests]
    x = np.arange(len(opponents))
    width = 0.8 / len(manifests)
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for index, manifest in enumerate(manifests):
        counts = manifest["opponent_counts"]
        axis.bar(
            x + index * width,
            [counts.get(name, 0) for name in opponents],
            width,
            label=splits[index],
        )
    axis.set_xticks(x + width * (len(manifests) - 1) / 2, opponents, rotation=20)
    axis.set_ylabel("transitions")
    axis.set_title("M2 demonstration distribution by opponent")
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=150,
        metadata={"Software": "BotColosseo", "CreationTime": None},
    )
    plt.close(figure)
    return output_path
