from __future__ import annotations

import io
import json
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from botcolosseo.agents.extraction_teachers import (
    ExtractionStyle,
    StyledExtractionTeacher,
)
from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_types import ExtractionActorObservation
from botcolosseo.envs.ipc import WorkerTimeout
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv

EXTRACTION_SCALAR_DIM = 9
EXTRACTION_DEMONSTRATION_FIELDS = (
    "frame",
    "scalars",
    "previous_action",
    "teacher_action",
    "episode_start",
    "valid_mask",
    "style_id",
    "train_seed",
)
STYLE_IDS = {
    style: index
    for index, style in enumerate(
        (
            ExtractionStyle.STRONG,
            ExtractionStyle.AGGRESSIVE,
            ExtractionStyle.DEFENSIVE,
            ExtractionStyle.EXPLORER,
        )
    )
}


class ExtractionRolloutController(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: ExtractionActorObservation) -> int: ...


@dataclass(frozen=True)
class ExtractionCase:
    split: str
    seed: int
    learner_side: str
    opponent_style: str
    layout_id: str = "base"

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "heldout", "test"}:
            raise ValueError(f"Invalid extraction split: {self.split}")
        if self.seed < 0:
            raise ValueError("Extraction case seed must be nonnegative")
        if self.learner_side not in {"host", "opponent"}:
            raise ValueError(f"Invalid extraction learner side: {self.learner_side}")
        if self.layout_id not in {"base", "heldout-a"}:
            raise ValueError(f"Invalid extraction layout: {self.layout_id}")
        ExtractionStyle(self.opponent_style)


def extraction_scalars(observation: ExtractionActorObservation) -> np.ndarray:
    return np.asarray(
        (
            observation.health / 100.0,
            observation.ammo / 40.0,
            observation.carried_value / 150.0,
            observation.free_slots / 3.0,
            observation.minimum_slot_value / 50.0,
            observation.banked_value / 150.0,
            float(observation.extraction_open),
            observation.extraction_progress,
            observation.remaining_time / 75.0,
        ),
        dtype=np.float32,
    )


class ExtractionDemonstrationBuffer:
    def __init__(self) -> None:
        self._rows: list[tuple[object, ...]] = []

    def __len__(self) -> int:
        return len(self._rows)

    def append(
        self,
        observation: ExtractionActorObservation,
        *,
        teacher_action: int,
        episode_start: bool,
        style: ExtractionStyle,
        train_seed: int,
    ) -> None:
        self._rows.append(
            (
                np.array(observation.frame, copy=True),
                extraction_scalars(observation),
                observation.previous_action,
                teacher_action,
                episode_start,
                True,
                STYLE_IDS[style],
                train_seed,
            )
        )

    def extend(self, other: ExtractionDemonstrationBuffer, *, limit: int) -> None:
        if limit < 0:
            raise ValueError("Extraction demonstration limit must be nonnegative")
        self._rows.extend(other._rows[:limit])

    def arrays(self) -> dict[str, np.ndarray]:
        if not self._rows:
            raise ValueError("Cannot materialize empty extraction demonstrations")
        columns = tuple(zip(*self._rows, strict=True))
        dtypes = (
            np.uint8,
            np.float32,
            np.int8,
            np.int8,
            np.bool_,
            np.bool_,
            np.uint8,
            np.int64,
        )
        arrays = {
            name: np.asarray(column, dtype=dtype)
            for name, column, dtype in zip(
                EXTRACTION_DEMONSTRATION_FIELDS,
                columns,
                dtypes,
                strict=True,
            )
        }
        validate_extraction_shard(arrays)
        return arrays


def validate_extraction_shard(arrays: dict[str, np.ndarray]) -> int:
    if set(arrays) != set(EXTRACTION_DEMONSTRATION_FIELDS):
        raise ValueError("Extraction demonstration fields do not match schema")
    lengths = {np.asarray(value).shape[0] for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("Extraction demonstration lengths are inconsistent")
    size = lengths.pop()
    if size <= 0:
        raise ValueError("Extraction demonstration shard must not be empty")
    expected = {
        "frame": ((size, 84, 84), np.dtype(np.uint8)),
        "scalars": ((size, EXTRACTION_SCALAR_DIM), np.dtype(np.float32)),
        "previous_action": ((size,), np.dtype(np.int8)),
        "teacher_action": ((size,), np.dtype(np.int8)),
        "episode_start": ((size,), np.dtype(np.bool_)),
        "valid_mask": ((size,), np.dtype(np.bool_)),
        "style_id": ((size,), np.dtype(np.uint8)),
        "train_seed": ((size,), np.dtype(np.int64)),
    }
    for name, (shape, dtype) in expected.items():
        array = np.asarray(arrays[name])
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(
                f"Invalid extraction {name}: {array.shape}/{array.dtype}"
            )
    if not bool(arrays["episode_start"][0]):
        raise ValueError("Extraction shards must start at an episode boundary")
    if not bool(np.all(arrays["valid_mask"])):
        raise ValueError("Extraction demonstrations must be fully supervised")
    if np.any(arrays["previous_action"] < 0) or np.any(
        arrays["previous_action"] >= 13
    ):
        raise ValueError("Extraction previous action is outside action space")
    if np.any(arrays["teacher_action"] < 0) or np.any(
        arrays["teacher_action"] >= 13
    ):
        raise ValueError("Extraction teacher action is outside action space")
    if np.any(arrays["style_id"] >= len(STYLE_IDS)):
        raise ValueError("Extraction style ID is outside style set")
    scalars = arrays["scalars"]
    if not np.isfinite(scalars).all() or np.any(scalars < 0) or np.any(scalars > 1):
        raise ValueError("Extraction Actor scalars must be normalized")
    return size


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def write_extraction_shard(
    arrays: dict[str, np.ndarray], output_path: Path
) -> Path:
    validate_extraction_shard(arrays)
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in EXTRACTION_DEMONSTRATION_FIELDS:
                info = zipfile.ZipInfo(
                    f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 0
                archive.writestr(
                    info,
                    _npy_bytes(arrays[name]),
                    compresslevel=9,
                )
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output_path


def load_extraction_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    validate_extraction_shard(arrays)
    return arrays


def load_extraction_cases(
    path: Path, *, expected_split: str
) -> tuple[ExtractionCase, ...]:
    if expected_split == "test" or path.name == "test.json":
        raise ValueError("Extraction demonstration generation cannot access test cases")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != expected_split:
        raise ValueError("Extraction case manifest has the wrong split")
    cases = tuple(ExtractionCase(**item) for item in payload["cases"])
    if not cases or any(case.split != expected_split for case in cases):
        raise ValueError("Extraction cases do not match requested split")
    if {case.learner_side for case in cases} != {"host", "opponent"}:
        raise ValueError("Extraction cases must cover both learner sides")
    return cases


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _collect_episode_once(
    *,
    root: Path,
    case: ExtractionCase,
    style: ExtractionStyle,
    max_decisions: int,
    rollout_controller: ExtractionRolloutController | None = None,
) -> tuple[ExtractionDemonstrationBuffer, dict[str, int]]:
    env = SynchronousExtractionEnv(
        config_path=root
        / "assets/scenarios/crystal_run_extraction/crystal_run_extraction.cfg",
        seed=case.seed,
        max_decisions=max_decisions,
    )
    learner = StyledExtractionTeacher(side=case.learner_side, style=style)
    opponent_side = "opponent" if case.learner_side == "host" else "host"
    opponent = StyledExtractionTeacher(
        side=opponent_side,
        style=ExtractionStyle(case.opponent_style),
    )
    buffer = ExtractionDemonstrationBuffer()
    event_counts: Counter[str] = Counter()
    try:
        observations, _ = env.reset()
        learner.reset()
        opponent.reset()
        if rollout_controller is not None:
            rollout_controller.reset()
        for decision in range(max_decisions):
            state = env.privileged_state()
            learner_observation = (
                observations.host
                if case.learner_side == "host"
                else observations.opponent
            )
            teacher_action = learner.act(state)
            learner_action = (
                teacher_action
                if rollout_controller is None
                else MacroAction(rollout_controller.act(learner_observation))
            )
            opponent_action = opponent.act(state)
            buffer.append(
                learner_observation,
                teacher_action=int(teacher_action),
                episode_start=decision == 0,
                style=style,
                train_seed=case.seed,
            )
            host_action, away_action = (
                (learner_action, opponent_action)
                if case.learner_side == "host"
                else (opponent_action, learner_action)
            )
            step = env.step(host_action, away_action)
            observations = type(observations)(step.host, step.opponent)
            event_counts.update(
                f"{event.side}:{event.type.value}" for event in step.events
            )
            learner_health = (
                step.host.health
                if case.learner_side == "host"
                else step.opponent.health
            )
            learner_banked = (
                step.host.banked_value
                if case.learner_side == "host"
                else step.opponent.banked_value
            )
            if (
                learner_health <= 0
                or learner_banked > 0
                or step.terminated
                or step.truncated
            ):
                break
        return buffer, dict(sorted(event_counts.items()))
    finally:
        env.close()


def _collect_episode(
    *,
    root: Path,
    case: ExtractionCase,
    style: ExtractionStyle,
    max_decisions: int,
    startup_attempts: int = 3,
    rollout_controller: ExtractionRolloutController | None = None,
) -> tuple[ExtractionDemonstrationBuffer, dict[str, int]]:
    if startup_attempts <= 0:
        raise ValueError("Extraction startup attempts must be positive")
    for attempt in range(startup_attempts):
        try:
            return _collect_episode_once(
                root=root,
                case=case,
                style=style,
                max_decisions=max_decisions,
                rollout_controller=rollout_controller,
            )
        except WorkerTimeout:
            if attempt + 1 == startup_attempts:
                raise
    raise AssertionError("Extraction startup retry loop did not return")


def generate_extraction_demonstrations(
    *,
    root: Path,
    split: str,
    cases_path: Path,
    output_dir: Path,
    style: ExtractionStyle | str,
    transitions: int,
    shard_size: int,
    max_decisions: int = 700,
    rollout_controller: ExtractionRolloutController | None = None,
    source_policy_sha256: str | None = None,
) -> dict[str, Any]:
    if transitions <= 0 or shard_size <= 0 or max_decisions <= 0:
        raise ValueError("Extraction generation sizes must be positive")
    if (rollout_controller is None) != (source_policy_sha256 is None):
        raise ValueError(
            "Correction generation requires both rollout policy and source hash"
        )
    selected_style = ExtractionStyle(style)
    cases = load_extraction_cases(cases_path, expected_split=split)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Extraction data output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_manifest = json.loads(
        (
            root
            / "assets/scenarios/crystal_run_extraction/manifest.json"
        ).read_text(encoding="utf-8")
    )
    shards: list[dict[str, object]] = []
    buffer = ExtractionDemonstrationBuffer()
    event_counts: Counter[str] = Counter()
    episode_count = 0
    case_index = 0

    def flush() -> None:
        nonlocal buffer
        if not len(buffer):
            return
        filename = f"{split}-{len(shards):05d}.npz"
        path = write_extraction_shard(buffer.arrays(), output_dir / filename)
        shards.append(
            {
                "file": filename,
                "sha256": sha256_file(path),
                "transitions": len(buffer),
            }
        )
        buffer = ExtractionDemonstrationBuffer()

    total = 0
    while total < transitions:
        case = cases[case_index % len(cases)]
        case_index += 1
        episode, counts = _collect_episode(
            root=root,
            case=case,
            style=selected_style,
            max_decisions=max_decisions,
            rollout_controller=rollout_controller,
        )
        if len(buffer) and len(buffer) + len(episode) > shard_size:
            flush()
        take = min(len(episode), transitions - total)
        buffer.extend(episode, limit=take)
        total += take
        episode_count += 1
        event_counts.update(counts)
        if len(buffer) >= shard_size or total == transitions:
            flush()

    manifest = {
        "case_manifest": str(cases_path.relative_to(root)),
        "case_manifest_sha256": sha256_file(cases_path),
        "episode_count": episode_count,
        "event_counts": dict(sorted(event_counts.items())),
        "scenario_hash": scenario_manifest["wad_sha256"],
        "schema_version": 1,
        "shards": shards,
        "split": split,
        "style": selected_style.value,
        "test_cases_accessed": False,
        "transitions": total,
        "generation_kind": (
            "teacher-rollout"
            if rollout_controller is None
            else "dagger-correction"
        ),
        "source_policy_sha256": source_policy_sha256,
    }
    manifest_path = output_dir / f"{split}-manifest.json"
    _atomic_json(manifest, manifest_path)
    return manifest
