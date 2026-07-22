# M4 Showcase Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable one-command pipeline that records matched real ViZDoom episodes and turns a Strong Base/Aggressive pair into GitHub-ready MP4, GIF, metrics, and hash-bound evidence without making premature milestone claims.

**Architecture:** Strict YAML and frozen validation-case manifests feed a sequential recorder that reuses the existing duel runtime and checkpoint loaders. Episode summaries drive deterministic case/highlight selection; media are staged, audited, and published with the manifest last. Development mode exercises the same path with M2 PPO/BC while being structurally forbidden from writing public M4 artifacts.

**Tech Stack:** Python 3.10, PyTorch 2.6, ViZDoom 1.3, NumPy, OpenCV, imageio/FFmpeg, PyYAML, Matplotlib, pytest, Ruff.

## Global Constraints

- Follow `docs/superpowers/specs/2026-07-22-m4-showcase-foundation-design.md`.
- Work only in `/home/wencong/BotColosseo/.worktrees/m4-showcase-foundation` on `feat/m4-showcase-foundation`.
- Do not modify the running M3 worktree or its artifacts.
- Runtime access is validation-only. Never open M2/M3 test or held-out manifests.
- M4 publication requires exactly `strong_base` and `aggressive`; M5 requires those plus `defensive` and `explorer`; development requires exactly `ppo` and `bc`.
- Development output stays under ignored `artifacts/showcase-development/`.
- Production output is restricted to `docs/assets/showcase/` and `reports/showcase/m4/`.
- Production binds checkpoint, scenario, case, config, metric, episode-log, and media SHA-256 values.
- GIF output is 10 FPS, 18 seconds, and at most `10_000_000` bytes.
- Never fall back to a script, random policy, different seed, stale media, or another checkpoint.
- Existing M1/M2 media and behavior remain unchanged.
- Use `PYTHONPATH=src /home/wencong/miniconda3/envs/botcolosseo/bin/python`.
- Every task follows red-green-refactor and ends in one focused commit.

---

## File Map

**Create**

- `configs/showcase/development-validation.json`: one frozen M2 validation case.
- `configs/showcase/m4-validation.json`: eight frozen M3 validation rows assigned to four script opponents.
- `configs/showcase/development.yaml`: non-public PPO/BC configuration.
- `src/botcolosseo/evaluation/showcase.py`: schemas, provenance, selectors, manifests, publication.
- `src/botcolosseo/demo/showcase.py`: overlay, episode capture, checkpoint adapter, metrics card.
- `src/botcolosseo/cli/render_showcase.py`: orchestration.
- `scripts/render_showcase.py`: thin entry point.
- `tests/unit/test_showcase_evaluation.py`
- `tests/unit/test_showcase_demo.py`
- `tests/unit/test_render_showcase_cli.py`
- `tests/integration/test_showcase_smoke.py`

**Modify**

- `src/botcolosseo/envs/video.py`: GIF writing and video-frame reading.
- `tests/unit/test_video.py`
- `tests/unit/test_public_docs.py`
- `script.md`
- `docs/superpowers/specs/2026-07-22-m4-showcase-foundation-design.md`

---

### Task 1: Freeze the Configuration and Case Provenance Contract

**Files:**
- Create: `configs/showcase/development-validation.json`
- Create: `configs/showcase/m4-validation.json`
- Create: `configs/showcase/development.yaml`
- Create: `src/botcolosseo/evaluation/showcase.py`
- Test: `tests/unit/test_showcase_evaluation.py`

**Interfaces:**
- Consumes: `DuelCase`, `DUEL_OPPONENTS`.
- Produces:
  - `ShowcasePolicySpec(policy_id, label, checkpoint, expected_sha256)`
  - `ShowcaseRenderSpec(fps, gif_seconds, gif_max_bytes, max_decisions)`
  - `ShowcaseConfig`
  - `case_id(case: DuelCase) -> str`
  - `load_showcase_config(path: Path, *, root: Path) -> ShowcaseConfig`
  - `load_showcase_cases(path: Path, *, root: Path, expected_count: int) -> tuple[DuelCase, ...]`

- [ ] **Step 1: Write failing schema and provenance tests**

Create `tests/unit/test_showcase_evaluation.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from botcolosseo.evaluation.showcase import (
    case_id,
    load_showcase_cases,
    load_showcase_config,
)


def test_development_config_is_non_public_and_hash_bound() -> None:
    config = load_showcase_config(
        Path("configs/showcase/development.yaml"), root=Path.cwd()
    )

    assert config.stage == "development"
    assert config.publication is False
    assert [policy.policy_id for policy in config.policies] == ["ppo", "bc"]
    assert config.metrics_path is None
    assert config.render.gif_max_bytes == 10_000_000
    assert config.output_dir == Path.cwd() / "artifacts/showcase-development/media"


def test_m4_cases_are_validation_only_and_cover_four_scripts() -> None:
    cases = load_showcase_cases(
        Path("configs/showcase/m4-validation.json"),
        root=Path.cwd(),
        expected_count=8,
    )

    assert {case.split for case in cases} == {"validation"}
    assert {case.learner_side for case in cases} == {"host", "opponent"}
    assert {case.opponent for case in cases} == {
        "fixed_route",
        "objective_first",
        "aggressive_script",
        "defensive_script",
    }
    assert len({case_id(case) for case in cases}) == 8


def test_case_manifest_rejects_source_hash_drift(tmp_path: Path) -> None:
    (tmp_path / "validation.json").write_text("[]\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "source": "validation.json",
        "source_sha256": "0" * 64,
        "cases": [],
    }
    path = tmp_path / "showcase.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="source hash"):
        load_showcase_cases(path, root=tmp_path, expected_count=0)


def test_publication_config_rejects_test_split(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "stage": "m4",
        "publication": True,
        "split": "test",
        "cases": "configs/showcase/m4-validation.json",
        "metrics": "reports/m4/validation/summary.json",
        "policies": [
            {
                "id": "strong_base",
                "label": "Strong Base",
                "checkpoint": "runs/m3/selected.pt",
                "expected_sha256": "1" * 64,
            },
            {
                "id": "aggressive",
                "label": "Aggressive",
                "checkpoint": "runs/m4/aggressive/selected.pt",
                "expected_sha256": "2" * 64,
            },
        ],
        "render": {
            "fps": 10,
            "gif_seconds": 18,
            "gif_max_bytes": 10_000_000,
            "max_decisions": 525,
            "output_dir": "docs/assets/showcase",
        },
        "evidence_dir": "reports/showcase/m4",
    }
    path = tmp_path / "m4.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="validation"):
        load_showcase_config(path, root=Path.cwd())
```

- [ ] **Step 2: Run red tests**

```bash
PYTHONPATH=src /home/wencong/miniconda3/envs/botcolosseo/bin/python \
  -m pytest tests/unit/test_showcase_evaluation.py -q
```

Expected: collection fails because `botcolosseo.evaluation.showcase` does not exist.

- [ ] **Step 3: Add the exact frozen manifests**

`configs/showcase/development-validation.json`:

```json
{
  "cases": [
    {
      "core_spawn_index": 0,
      "learner_side": "host",
      "opponent": "random_legal",
      "pair_index": 250,
      "route": "direct_lower",
      "seed": 656489971,
      "split": "validation"
    }
  ],
  "schema_version": 1,
  "source": "configs/m2/validation.json",
  "source_sha256": "3f5d0c4f6ac26541c6d159f53413f7c1c1cc44702c49159983a84a2ff5270993"
}
```

`configs/showcase/m4-validation.json`:

```json
{
  "cases": [
    {"core_spawn_index": 1, "learner_side": "host", "opponent": "fixed_route", "pair_index": 250, "route": "direct_upper", "seed": 1019181764, "split": "validation"},
    {"core_spawn_index": 1, "learner_side": "opponent", "opponent": "fixed_route", "pair_index": 250, "route": "direct_upper", "seed": 1019181764, "split": "validation"},
    {"core_spawn_index": 2, "learner_side": "host", "opponent": "objective_first", "pair_index": 251, "route": "direct_lower", "seed": 245721494, "split": "validation"},
    {"core_spawn_index": 2, "learner_side": "opponent", "opponent": "objective_first", "pair_index": 251, "route": "direct_lower", "seed": 245721494, "split": "validation"},
    {"core_spawn_index": 0, "learner_side": "host", "opponent": "aggressive_script", "pair_index": 252, "route": "flank", "seed": 1891682600, "split": "validation"},
    {"core_spawn_index": 0, "learner_side": "opponent", "opponent": "aggressive_script", "pair_index": 252, "route": "flank", "seed": 1891682600, "split": "validation"},
    {"core_spawn_index": 1, "learner_side": "host", "opponent": "defensive_script", "pair_index": 253, "route": "direct_lower", "seed": 1200430541, "split": "validation"},
    {"core_spawn_index": 1, "learner_side": "opponent", "opponent": "defensive_script", "pair_index": 253, "route": "direct_lower", "seed": 1200430541, "split": "validation"}
  ],
  "schema_version": 1,
  "source": "configs/m3/validation.json",
  "source_sha256": "1af280329241e244a6a83a143854c3eeb53f60a52f7f5fc986e4df88c40a8eb2"
}
```

`configs/showcase/development.yaml`:

```yaml
schema_version: 1
stage: development
publication: false
split: validation
cases: configs/showcase/development-validation.json
metrics: null
policies:
  - id: ppo
    label: PPO
    checkpoint: runs/m2/ppo-full/selected.pt
    expected_sha256: 41bc3040faf260f928d2b14cf970d6e338fc56f3d80d0df50742b774d51d6647
  - id: bc
    label: BC
    checkpoint: runs/m2/bc-full/best.pt
    expected_sha256: 7eef23a06ea7177d5090ba90be65f8f2f1a847ecb15d81035c21a7e4567949d4
render:
  fps: 10
  gif_seconds: 18
  gif_max_bytes: 10000000
  max_decisions: 525
  output_dir: artifacts/showcase-development/media
evidence_dir: artifacts/showcase-development/evidence
```

- [ ] **Step 4: Implement strict dataclasses and parsing**

In `src/botcolosseo/evaluation/showcase.py`, use exact field sets and reject unknown keys. The core validation is:

```python
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONFIG_FIELDS = {
    "schema_version", "stage", "publication", "split", "cases", "metrics",
    "policies", "render", "evidence_dir",
}
_POLICY_FIELDS = {"id", "label", "checkpoint", "expected_sha256"}
_RENDER_FIELDS = {
    "fps", "gif_seconds", "gif_max_bytes", "max_decisions", "output_dir"
}
_CASE_FIELDS = {
    "split", "pair_index", "seed", "opponent", "learner_side",
    "core_spawn_index", "route",
}


@dataclass(frozen=True)
class ShowcasePolicySpec:
    policy_id: str
    label: str
    checkpoint: Path
    expected_sha256: str

    def __post_init__(self) -> None:
        if not self.policy_id or not self.label:
            raise ValueError("Showcase policy ID and label must be non-empty")
        if _SHA256.fullmatch(self.expected_sha256) is None:
            raise ValueError("Showcase policy requires a lowercase SHA-256")


@dataclass(frozen=True)
class ShowcaseRenderSpec:
    fps: int
    gif_seconds: int
    gif_max_bytes: int
    max_decisions: int

    def __post_init__(self) -> None:
        if not 0 < self.fps <= 30:
            raise ValueError("Showcase FPS must be in [1, 30]")
        if not 0 < self.gif_seconds <= 20:
            raise ValueError("Showcase GIF must be at most 20 seconds")
        if self.gif_max_bytes != 10_000_000:
            raise ValueError("Showcase GIF ceiling must remain 10000000 bytes")
        if self.max_decisions <= 0:
            raise ValueError("Showcase max decisions must be positive")


@dataclass(frozen=True)
class ShowcaseConfig:
    stage: str
    publication: bool
    split: str
    cases_path: Path
    metrics_path: Path | None
    policies: tuple[ShowcasePolicySpec, ...]
    render: ShowcaseRenderSpec
    output_dir: Path
    evidence_dir: Path
    config_path: Path
    config_sha256: str


def case_id(case: DuelCase) -> str:
    return f"{case.opponent}:{case.pair_index}:{case.learner_side}"
```

`load_showcase_config` must enforce:

- schema version 1 and `split == "validation"`;
- ordered IDs `["ppo", "bc"]` for development, `["strong_base", "aggressive"]` for M4, and `["strong_base", "aggressive", "defensive", "explorer"]` for M5;
- `metrics is None` only for development;
- public media root `docs/assets/showcase`, stage-specific evidence root `reports/showcase/<stage>`, and the ignored development root exactly as defined globally;
- path resolution relative to repository `root`;
- config SHA-256 from the original bytes.

Use one explicit stage map rather than scattered conditionals:

```python
expected_policy_ids = {
    "development": ("ppo", "bc"),
    "m4": ("strong_base", "aggressive"),
    "m5": ("strong_base", "aggressive", "defensive", "explorer"),
}
```

Reject every other stage. Require `publication is False` only for development
and `publication is True` for M4/M5.

`load_showcase_cases` must:

1. verify the source file SHA-256 before parsing cases;
2. require the exact row count;
3. require each row to be validation-only and in `DUEL_OPPONENTS`;
4. prove every field except the assigned `opponent` exists in the source M3 row;
5. when the source already has `opponent` (M2 development), require it to match;
6. reject duplicate `case_id` values.

- [ ] **Step 5: Run green tests and lint**

```bash
PYTHONPATH=src /home/wencong/miniconda3/envs/botcolosseo/bin/python \
  -m pytest tests/unit/test_showcase_evaluation.py -q
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check \
  src/botcolosseo/evaluation/showcase.py tests/unit/test_showcase_evaluation.py
```

Expected: all focused tests pass; Ruff reports `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add configs/showcase src/botcolosseo/evaluation/showcase.py \
  tests/unit/test_showcase_evaluation.py
git commit -m "feat: freeze the showcase publication contract"
```

---

### Task 2: Add Atomic GIF I/O and Learner-Perspective Composition

**Files:**
- Modify: `src/botcolosseo/envs/video.py:1-55`
- Create: `src/botcolosseo/demo/showcase.py`
- Modify: `tests/unit/test_video.py`
- Create: `tests/unit/test_showcase_demo.py`

**Interfaces:**
- Produces:
  - `write_gif(frames, output_path, *, fps, max_bytes) -> Path`
  - `read_video_frames(path) -> tuple[NDArray[np.uint8], ...]`
  - `compose_learner_frame(observation, *, policy_label, event_label)`
  - `compose_showcase_comparison(streams, *, subtitle)`

- [ ] **Step 1: Write red tests**

Append to `tests/unit/test_video.py`:

```python
from botcolosseo.envs.video import read_video_frames, write_gif


def test_write_gif_is_atomic_and_enforces_size(tmp_path: Path, monkeypatch) -> None:
    appended: list[np.ndarray] = []

    class FakeWriter:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def append_data(self, frame: np.ndarray) -> None:
            appended.append(frame)

    def fake_get_writer(path: Path, **kwargs):
        Path(path).write_bytes(b"GIF89a")
        return FakeWriter()

    monkeypatch.setattr("botcolosseo.envs.video.imageio.get_writer", fake_get_writer)
    target = tmp_path / "comparison.gif"

    result = write_gif(
        [np.zeros((8, 8), dtype=np.uint8)],
        target,
        fps=10,
        max_bytes=10,
    )

    assert result == target
    assert target.read_bytes() == b"GIF89a"
    assert len(appended) == 1


def test_read_video_frames_normalizes_rgba(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "botcolosseo.envs.video.imageio.get_reader",
        lambda path: iter([np.zeros((5, 7, 4), dtype=np.uint8)]),
    )

    frames = read_video_frames(tmp_path / "episode.mp4")

    assert len(frames) == 1
    assert frames[0].shape == (5, 7, 3)
```

Create `tests/unit/test_showcase_demo.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from botcolosseo.demo.showcase import (
    compose_learner_frame,
    compose_showcase_comparison,
)
from botcolosseo.envs.duel_types import DuelActorObservation


def observation() -> DuelActorObservation:
    return DuelActorObservation(
        frame=np.full((84, 84), 80, dtype=np.uint8),
        health=100.0,
        armor=0.0,
        ammo=10.0,
        own_score=1,
        opponent_score=0,
        has_core=True,
        previous_action=0,
    )


def test_learner_frame_has_fixed_rgb_geometry() -> None:
    frame = compose_learner_frame(
        observation(), policy_label="Aggressive", event_label="VALID_HIT"
    )

    assert frame.shape == (300, 256, 3)
    assert frame.dtype == np.uint8
    assert frame[48:].mean() > 0


def test_comparison_aligns_unequal_streams() -> None:
    base = [np.full((300, 256, 3), 10, dtype=np.uint8)]
    aggressive = [
        np.full((300, 256, 3), value, dtype=np.uint8) for value in (20, 30)
    ]

    frames = compose_showcase_comparison(
        (("Strong Base", base), ("Aggressive", aggressive)),
        subtitle="VALIDATION | seed=7 | vs fixed_route | host",
    )

    assert len(frames) == 2
    assert frames[0].shape == (332, 512, 3)
    assert np.array_equal(frames[1][-300:, :256], base[0])


def test_comparison_rejects_empty_stream() -> None:
    valid = [np.zeros((300, 256, 3), dtype=np.uint8)]
    with pytest.raises(ValueError, match="empty"):
        compose_showcase_comparison(
            (("Strong Base", valid), ("Aggressive", [])),
            subtitle="validation",
        )
```

- [ ] **Step 2: Verify red state**

Run both files. Expected: imports fail for the new interfaces.

- [ ] **Step 3: Implement GIF and video reads**

Add to `envs/video.py`:

```python
def write_gif(
    frames: Iterable[NDArray[np.generic]],
    output_path: Path,
    *,
    fps: int,
    max_bytes: int,
) -> Path:
    if fps <= 0 or max_bytes <= 0:
        raise ValueError("GIF fps and max_bytes must be positive")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    try:
        with imageio.get_writer(
            temporary,
            format="GIF",
            mode="I",
            duration=1000.0 / fps,
            loop=0,
        ) as writer:
            count = 0
            for frame in frames:
                writer.append_data(normalize_rgb_frame(frame))
                count += 1
        if count == 0:
            raise ValueError("Cannot write an empty GIF")
        if temporary.stat().st_size > max_bytes:
            raise ValueError("Showcase GIF exceeds the configured byte ceiling")
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output_path


def read_video_frames(path: Path) -> tuple[NDArray[np.uint8], ...]:
    frames = tuple(normalize_rgb_frame(frame) for frame in imageio.get_reader(path))
    if not frames:
        raise ValueError("Cannot read an empty video")
    return frames
```

- [ ] **Step 4: Implement fixed visual geometry**

Create `demo/showcase.py` with:

```python
def compose_learner_frame(
    observation: DuelActorObservation,
    *,
    policy_label: str,
    event_label: str,
) -> NDArray[np.uint8]:
    if not policy_label or not event_label:
        raise ValueError("Showcase overlay labels must be non-empty")
    view = cv2.resize(observation.frame, (256, 252), interpolation=cv2.INTER_NEAREST)
    view = cv2.cvtColor(view, cv2.COLOR_GRAY2RGB)
    canvas = np.zeros((300, 256, 3), dtype=np.uint8)
    canvas[48:] = view
    cv2.putText(
        canvas, policy_label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )
    state = (
        f"score={observation.own_score}-{observation.opponent_score} "
        f"core={int(observation.has_core)} event={event_label}"
    )
    cv2.putText(
        canvas, state, (6, 38), cv2.FONT_HERSHEY_SIMPLEX,
        0.34, (220, 220, 220), 1, cv2.LINE_AA,
    )
    return canvas


def compose_showcase_comparison(
    streams: Sequence[tuple[str, Sequence[NDArray[np.uint8]]]],
    *,
    subtitle: str,
) -> tuple[NDArray[np.uint8], ...]:
    if len(streams) < 2 or any(not label or not frames for label, frames in streams):
        raise ValueError("Showcase comparison contains an empty stream")
    shape = np.asarray(streams[0][1][0]).shape
    if shape != (300, 256, 3) or any(
        np.asarray(frame).shape != shape
        for _, frames in streams
        for frame in frames
    ):
        raise ValueError("Showcase comparison streams have incompatible geometry")
    result = []
    for index in range(max(len(frames) for _, frames in streams)):
        panels = [
            np.array(frames[min(index, len(frames) - 1)], copy=True)
            for _, frames in streams
        ]
        comparison = np.concatenate(panels, axis=1)
        canvas = np.zeros((332, comparison.shape[1], 3), dtype=np.uint8)
        canvas[32:] = comparison
        cv2.putText(
            canvas, subtitle, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (255, 255, 255), 1, cv2.LINE_AA,
        )
        result.append(canvas)
    return tuple(result)
```

- [ ] **Step 5: Green verification**

```bash
PYTHONPATH=src /home/wencong/miniconda3/envs/botcolosseo/bin/python \
  -m pytest tests/unit/test_video.py tests/unit/test_showcase_demo.py \
  tests/unit/test_m2_showcase.py -q
```

Expected: all pass; M2 showcase behavior is unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/botcolosseo/envs/video.py src/botcolosseo/demo/showcase.py \
  tests/unit/test_video.py tests/unit/test_showcase_demo.py
git commit -m "feat: add GitHub showcase visual primitives"
```

---

### Task 3: Record Auditable Learner Episodes

**Files:**
- Modify: `src/botcolosseo/demo/showcase.py`
- Modify: `tests/unit/test_showcase_demo.py`

**Interfaces:**
- Produces:
  - `ShowcaseEvent(decision_index, label)`
  - `RecordedShowcaseEpisode.to_record() -> dict[str, object]`
  - `CheckpointEvaluationPolicy`
  - `record_showcase_episode(case, *, policy_id, policy_label, policy, graph, config_path, max_decisions, environment_factory=None) -> RecordedShowcaseEpisode`

- [ ] **Step 1: Add red record/adapter tests**

```python
def test_recorded_episode_serializes_without_frames() -> None:
    episode = RecordedShowcaseEpisode(
        policy_id="strong_base",
        case=DuelCase("validation", 1, 7, "fixed_route", "host", 0, "flank"),
        frames=(np.zeros((300, 256, 3), dtype=np.uint8),),
        events=(ShowcaseEvent(0, "PICKUP"),),
        decisions=1,
        learner_score=1,
        opponent_score=0,
        objective_completed=True,
        terminated=True,
        truncated=False,
        peer_tic_lag_max=0,
        protocol_inconsistent=False,
        action_tic_inconsistent=False,
        score_event_inconsistent=False,
        environment_attempts=1,
        scenario_hash="a" * 64,
    )

    payload = episode.to_record()

    assert "frames" not in payload
    assert payload["case_id"] == "fixed_route:1:host"
    assert payload["events"] == [{"decision_index": 0, "label": "PICKUP"}]


def test_checkpoint_adapter_does_not_use_privileged_state() -> None:
    calls: list[str] = []

    class FakePolicy:
        def reset(self) -> None:
            calls.append("reset")

        def act(self, actor_observation):
            calls.append("act")
            return 0

    adapter = CheckpointEvaluationPolicy("strong_base", FakePolicy())
    adapter.reset(seed=7)
    result = adapter.act(observation(), object())

    assert int(result) == 0
    assert calls == ["reset", "act"]
```

Expected red state: record and adapter imports do not exist.

- [ ] **Step 2: Implement record models and adapter**

```python
@dataclass(frozen=True)
class ShowcaseEvent:
    decision_index: int
    label: str


@dataclass(frozen=True)
class RecordedShowcaseEpisode:
    policy_id: str
    case: DuelCase
    frames: tuple[NDArray[np.uint8], ...]
    events: tuple[ShowcaseEvent, ...]
    decisions: int
    learner_score: int
    opponent_score: int
    objective_completed: bool
    terminated: bool
    truncated: bool
    peer_tic_lag_max: int
    protocol_inconsistent: bool
    action_tic_inconsistent: bool
    score_event_inconsistent: bool
    environment_attempts: int
    scenario_hash: str

    def to_record(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "case": self.case.to_dict(),
            "case_id": case_id(self.case),
            "events": [asdict(event) for event in self.events],
            "decisions": self.decisions,
            "learner_score": self.learner_score,
            "opponent_score": self.opponent_score,
            "objective_completed": self.objective_completed,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "peer_tic_lag_max": self.peer_tic_lag_max,
            "protocol_inconsistent": self.protocol_inconsistent,
            "action_tic_inconsistent": self.action_tic_inconsistent,
            "score_event_inconsistent": self.score_event_inconsistent,
            "environment_attempts": self.environment_attempts,
            "scenario_hash": self.scenario_hash,
        }


class CheckpointEvaluationPolicy:
    def __init__(self, name: str, policy: PublicCheckpointPolicy) -> None:
        self.name = name
        self._policy = policy

    def reset(self, *, seed: int) -> None:
        del seed
        self._policy.reset()

    def act(self, observation: DuelActorObservation, state: object) -> MacroAction:
        del state
        return MacroAction(self._policy.act(observation))
```

- [ ] **Step 3: Implement single-attempt capture**

Base the control loop on existing `record_showcase_episode` in `demo/m2_showcase.py`, but change the returned frames to learner-only overlays. Track:

- learner-side `PICKUP`, `VALID_HIT`, `DROP`, `SCORE`;
- score-event agreement;
- `peer_tic_lag_max`;
- terminal action-tic validity via `valid_action_tic_boundary`;
- `environment_attempts=1` with no retry wrapper;
- the reset `scenario_hash`.

Exact signature:

```python
def record_showcase_episode(
    case: DuelCase,
    *,
    policy_id: str,
    policy_label: str,
    policy: EvaluationPolicy,
    graph: RegionGraph,
    config_path: Path,
    max_decisions: int,
    environment_factory: EnvironmentFactory | None = None,
) -> RecordedShowcaseEpisode:
```

On every decision, append exactly one overlay frame. Raise if decisions exceed `max_decisions`; always close the environment in `finally`.

- [ ] **Step 4: Test capture using two real schema steps**

Construct a fake environment with two `DuelStep` values:

1. decision 0: learner `PICKUP`, score 0-0;
2. decision 1: learner `SCORE`, score 1-0, `terminated=True`.

Use real `DuelActorObservation`, `DuelEvent`, and `DuelEventType`. Assert:

```python
assert episode.decisions == 2
assert episode.objective_completed is True
assert [event.label for event in episode.events] == ["PICKUP", "SCORE"]
assert episode.peer_tic_lag_max == 0
assert episode.action_tic_inconsistent is False
assert episode.score_event_inconsistent is False
assert fake_environment.closed is True
```

- [ ] **Step 5: Green verification and commit**

```bash
PYTHONPATH=src /home/wencong/miniconda3/envs/botcolosseo/bin/python \
  -m pytest tests/unit/test_showcase_demo.py -q
git add src/botcolosseo/demo/showcase.py tests/unit/test_showcase_demo.py
git commit -m "feat: record auditable showcase episodes"
```

---

### Task 4: Validate Metrics and Select Cases/Highlights

**Files:**
- Modify: `src/botcolosseo/evaluation/showcase.py`
- Modify: `tests/unit/test_showcase_evaluation.py`

**Interfaces:**
- Produces:
  - `ShowcaseMetricEvidence`
  - `load_metric_evidence(path, *, expected_stage, expected_hashes)`
  - `ShowcaseSelection`
  - `select_showcase_case(records, policy_ids, contrast_scores)`
  - `select_highlight_window(scores, *, window_frames)`

- [ ] **Step 1: Write red metric and selector tests**

Use this frozen metric payload:

```python
payload = {
    "schema_version": 1,
    "stage": "m4",
    "split": "validation",
    "passed": True,
    "style_gate_passed": True,
    "retention_gate_passed": True,
    "episodes": 800,
    "checkpoint_sha256": {"strong_base": "1" * 64, "aggressive": "2" * 64},
    "headline": {
        "base_win_rate": 0.72,
        "aggressive_style_delta": 0.31,
        "skill_retention": 0.89,
    },
    "case_contrast_scores": {"fixed_route:250:host": 0.5},
    "decision_contrast_scores": {"fixed_route:250:host": [0.0, 1.0, 0.0]},
}
```

Tests must prove:

```python
assert evidence.skill_retention == 0.89
assert evidence.case_contrast_scores["fixed_route:250:host"] == 0.5
assert select_highlight_window(
    (0.0, 1.0, 1.0, 0.0, 1.0), window_frames=3
) == (0, 3)
```

Create two eligible cases with equal contrast and assert lexicographic case-ID tie-breaking. Create another case where one record is truncated and assert it is rejected. If every case is rejected, require `ValueError("No showcase case satisfies publication eligibility")`.

- [ ] **Step 2: Implement the exact metric schema**

```python
@dataclass(frozen=True)
class ShowcaseMetricEvidence:
    episodes: int
    base_win_rate: float
    aggressive_style_delta: float
    skill_retention: float
    checkpoint_sha256: dict[str, str]
    case_contrast_scores: dict[str, float]
    decision_contrast_scores: dict[str, tuple[float, ...]]
    source_sha256: str
```

`load_metric_evidence` requires all three pass booleans, stage equality with the active `ShowcaseConfig`, split `validation`, positive episodes, checkpoint equality for every configured policy, base win/retention in `[0,1]`, positive style delta, identical case keys in case/decision score dictionaries, and nonempty decision vectors. Add a test proving an M5 four-policy hash map is accepted only when `expected_stage="m5"`.

- [ ] **Step 3: Implement hard eligibility**

A record is ineligible when any of these is true:

```python
record["terminated"] is not True
record["truncated"] is not False
record["objective_completed"] is not True
record["environment_attempts"] != 1
record["peer_tic_lag_max"] != 0
record["protocol_inconsistent"] is True
record["action_tic_inconsistent"] is True
record["score_event_inconsistent"] is True
```

Group by `case_id`; require every configured policy exactly once. Rank eligible cases by `(-contrast_score, case_id)`. Do not include skill retention in the rank because it is a gate.

- [ ] **Step 4: Implement deterministic highlight selection**

```python
def select_highlight_window(
    scores: Sequence[float], *, window_frames: int
) -> tuple[int, int]:
    if not scores or window_frames <= 0:
        raise ValueError("Highlight selection requires scores and a positive window")
    width = min(window_frames, len(scores))
    totals = [
        sum(scores[start : start + width])
        for start in range(len(scores) - width + 1)
    ]
    best = max(range(len(totals)), key=lambda index: (totals[index], -index))
    return best, best + width
```

- [ ] **Step 5: Green verification and commit**

```bash
PYTHONPATH=src /home/wencong/miniconda3/envs/botcolosseo/bin/python \
  -m pytest tests/unit/test_showcase_evaluation.py -q
git add src/botcolosseo/evaluation/showcase.py tests/unit/test_showcase_evaluation.py
git commit -m "feat: select honest showcase highlights"
```

---

### Task 5: Build Metrics, Manifest, and Manifest-Last Publication

**Files:**
- Modify: `src/botcolosseo/demo/showcase.py`
- Modify: `src/botcolosseo/evaluation/showcase.py`
- Modify: `tests/unit/test_showcase_demo.py`
- Modify: `tests/unit/test_showcase_evaluation.py`

**Interfaces:**
- Produces:
  - `render_metrics_card(evidence, output) -> Path`
  - `canonical_json(payload) -> bytes`
  - `write_jsonl(path, rows) -> Path`
  - `build_showcase_manifest(*, git_commit, git_dirty, config, scenario_hash, case_manifest_sha256, checkpoint_sha256, metric_sha256, episodes_path, selected_case, highlight, media, gate_passed) -> dict[str, object]`
  - `publish_staged_files(files, staged_manifest, target_manifest, run_identity)`

- [ ] **Step 1: Write red metrics/publication tests**

Metrics test:

```python
output = render_metrics_card(evidence, tmp_path / "metrics.png")
assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
assert output.stat().st_size > 10_000
```

Publication test:

```python
publish_staged_files(
    ((staged_media, target_media),),
    staged_manifest=staged_manifest,
    target_manifest=target_manifest,
    run_identity="a" * 64,
)
assert target_media.read_bytes() == b"GIF89a"
assert json.loads(target_manifest.read_text())["run_identity"] == "a" * 64
```

Then change the staged identity to `"b" * 64` and require rejection because the target manifest already commits identity `a`.

- [ ] **Step 2: Implement the four-number card**

Render exactly: Base win rate, Aggressive style shift, Skill retention, Episodes. Use one horizontal Matplotlib row, dark background, atomic temporary PNG, no free-form claims.

- [ ] **Step 3: Implement canonical evidence**

```python
def canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path
```

`build_showcase_manifest` must include all spec fields and compute:

```python
identity_payload = {
    "config_sha256": config.config_sha256,
    "scenario_hash": scenario_hash,
    "case_manifest_sha256": case_manifest_sha256,
    "checkpoint_sha256": checkpoint_sha256,
    "metric_sha256": metric_sha256,
    "selected_case": selected_case,
    "highlight": list(highlight),
}
run_identity = hashlib.sha256(canonical_json(identity_payload)).hexdigest()
```

Media rows contain relative path, SHA-256, bytes, frame count, dimensions, and FPS. Include `split="validation"`, `official_test_result=False`, and `test_cases_accessed=False`.

- [ ] **Step 4: Implement publication with manifest last**

If a target manifest exists with another `run_identity`, raise before copying. Copy each staged file to a temporary sibling and replace it. Copy the verified staged manifest last. The manifest is the only completion marker.

- [ ] **Step 5: Green verification and commit**

```bash
PYTHONPATH=src /home/wencong/miniconda3/envs/botcolosseo/bin/python \
  -m pytest tests/unit/test_showcase_demo.py tests/unit/test_showcase_evaluation.py -q
git add src/botcolosseo/demo/showcase.py src/botcolosseo/evaluation/showcase.py \
  tests/unit/test_showcase_demo.py tests/unit/test_showcase_evaluation.py
git commit -m "feat: publish hash-bound showcase evidence"
```

---

### Task 6: Orchestrate the One-Command Renderer

**Files:**
- Create: `src/botcolosseo/cli/render_showcase.py`
- Create: `scripts/render_showcase.py`
- Create: `tests/unit/test_render_showcase_cli.py`

**Interfaces:**
- Consumes all Task 1-5 interfaces plus existing `load_actor_policy` and `CheckpointOpponentPolicy.load`.
- Produces:
  - `build_parser()`
  - `load_showcase_policy(policy, *, publication, checkpoint_root, scenario_hash, device)`
  - `render_showcase(*, root, config_path, checkpoint_root, device) -> dict[str, object]`
  - `main(argv=None) -> int`

- [ ] **Step 1: Write red CLI surface tests**

```python
def test_cli_exposes_only_config_checkpoint_root_and_device() -> None:
    args = build_parser().parse_args(
        [
            "--config", "configs/showcase/development.yaml",
            "--checkpoint-root", "artifacts",
            "--device", "cpu",
        ]
    )

    assert args.config == Path("configs/showcase/development.yaml")
    assert args.checkpoint_root == Path("artifacts")
    assert args.device == "cpu"


def test_unknown_development_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="development policy"):
        load_showcase_policy(
            ShowcasePolicySpec("random", "Random", Path("x.pt"), "1" * 64),
            publication=False,
            checkpoint_root=Path.cwd(),
            scenario_hash="a" * 64,
            device="cpu",
        )
```

- [ ] **Step 2: Add thin script and parser**

`scripts/render_showcase.py`:

```python
#!/usr/bin/env python3
from botcolosseo.cli.render_showcase import main

if __name__ == "__main__":
    raise SystemExit(main())
```

Parser:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render auditable Bot showcase media")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cpu")
    return parser
```

- [ ] **Step 3: Delegate checkpoint loading**

For development IDs, verify the file hash and call existing `load_actor_policy`. For M4 IDs, build an `OpponentSpec`, call existing `CheckpointOpponentPolicy.load`, then wrap it with `CheckpointEvaluationPolicy`. Do not parse either checkpoint schema again.

- [ ] **Step 4: Implement sequential staging**

`render_showcase` must:

1. load config and 1 or 8 frozen cases;
2. load scenario hash, region graph, and policies; for publication load metric
   evidence with
   `load_metric_evidence(config.metrics_path, expected_stage=config.stage,
   expected_hashes={policy.policy_id: policy.expected_sha256 for policy in config.policies})`;
3. reject dirty worktree only for production;
4. use one temporary staging directory;
5. record one policy/case at a time, immediately write its MP4, then release raw frames;
6. write canonical episode JSONL;
7. run eligibility/ranking and highlight selection;
8. decode only selected MP4s, slice the same highlight range, compose GIF;
9. render metrics in M4/M5 publication mode and omit them in development mode;
10. build and verify the manifest;
11. publish media/evidence with manifest last.

Exact M4 target names:

```python
{
    "comparison_gif": config.output_dir / "m4-base-vs-aggressive.gif",
    "strong_base_mp4": config.output_dir / "m4-strong-base.mp4",
    "aggressive_mp4": config.output_dir / "m4-aggressive.mp4",
    "metrics_png": config.output_dir / "m4-metrics.png",
    "episodes": config.evidence_dir / "episodes.jsonl",
    "selection": config.evidence_dir / "case-selection.json",
    "manifest": config.evidence_dir / "manifest.json",
}
```

For M5, use `m5-style-comparison.gif`, `m5-strong-base.mp4`, `m5-aggressive.mp4`, `m5-defensive.mp4`, `m5-explorer.mp4`, and `m5-metrics.png`, with evidence under `reports/showcase/m5/`. Select names from `config.stage`; do not branch on policy model internals. Development target names are `development-comparison.gif`, `development-ppo.mp4`, and `development-bc.mp4` under the ignored config paths.

- [ ] **Step 5: Unit-test orchestration with fakes**

Monkeypatch policy loading, episode recording, video writing/reading, subprocess Git calls, and metric loading. Prove:

- development returns `publication=False` and never creates public directories;
- mismatched checkpoint hash fails before episode capture;
- production without passing metrics fails before capture;
- manifest is published after every referenced file;
- two unequal selected MP4 streams yield one aligned GIF.

- [ ] **Step 6: Run focused tests and commit**

```bash
PYTHONPATH=src /home/wencong/miniconda3/envs/botcolosseo/bin/python \
  -m pytest tests/unit/test_render_showcase_cli.py \
  tests/unit/test_showcase_demo.py tests/unit/test_showcase_evaluation.py \
  tests/unit/test_video.py -q
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check \
  src/botcolosseo/cli/render_showcase.py scripts/render_showcase.py \
  tests/unit/test_render_showcase_cli.py
git add src/botcolosseo/cli/render_showcase.py scripts/render_showcase.py \
  tests/unit/test_render_showcase_cli.py
git commit -m "feat: orchestrate reproducible showcase rendering"
```

---

### Task 7: Run the Real Development Smoke and Publish the Draft PR

**Files:**
- Create: `tests/integration/test_showcase_smoke.py`
- Modify: `tests/unit/test_public_docs.py`
- Modify: `script.md`

**Interfaces:**
- Consumes: complete renderer and local M2 checkpoint root.
- Produces: opt-in real smoke, runbook, verified remote branch, Draft PR.

- [ ] **Step 1: Write opt-in real smoke**

```python
@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("BOTCOLOSSEO_RUN_SHOWCASE_SMOKE") != "1",
    reason="set BOTCOLOSSEO_RUN_SHOWCASE_SMOKE=1 for real ViZDoom media",
)
def test_real_m2_showcase_writes_non_public_hash_bound_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path.cwd()
    checkpoint_root = Path(os.environ["BOTCOLOSSEO_M2_ARTIFACT_ROOT"])
    config = load_showcase_config(
        Path("configs/showcase/development.yaml"), root=root
    )
    isolated = replace(
        config,
        output_dir=tmp_path / "media",
        evidence_dir=tmp_path / "evidence",
    )
    monkeypatch.setattr(
        showcase_cli, "load_showcase_config", lambda path, root: isolated
    )

    manifest = showcase_cli.render_showcase(
        root=root,
        config_path=config.config_path,
        checkpoint_root=checkpoint_root,
        device="cuda:0",
    )

    assert manifest["publication"] is False
    assert manifest["split"] == "validation"
    assert manifest["official_test_result"] is False
    assert manifest["test_cases_accessed"] is False
    assert manifest["m4_gate_passed"] is False
    assert (tmp_path / "media/development-ppo.mp4").stat().st_size > 10_000
    assert (tmp_path / "media/development-bc.mp4").stat().st_size > 10_000
    assert (tmp_path / "media/development-comparison.gif").stat().st_size > 10_000
    saved = json.loads((tmp_path / "evidence/manifest.json").read_text())
    assert saved["run_identity"] == manifest["run_identity"]
```

- [ ] **Step 2: Guard public README truthfulness**

Append to `test_public_docs.py`:

```python
def test_readme_does_not_claim_m4_media_before_publication() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "m4-base-vs-aggressive.gif" not in readme
    assert "M4 passed" not in readme
```

- [ ] **Step 3: Run CPU gate and skipped smoke**

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /home/wencong/miniconda3/envs/botcolosseo/bin/python \
  -m pytest tests/unit tests/integration/test_showcase_smoke.py -q
```

Expected: unit suite passes; real smoke skips with the explicit reason.

- [ ] **Step 4: Run real smoke interactively**

Check GPU processes first. Do not use `nohup`. Prefer a GPU not occupied by M3; a cross-play evaluator using under 2 GiB may share GPU 1, but a trainer may not.

```bash
cd /home/wencong/BotColosseo/.worktrees/m4-showcase-foundation
env \
  CUDA_VISIBLE_DEVICES=1 \
  BOTCOLOSSEO_RUN_SHOWCASE_SMOKE=1 \
  BOTCOLOSSEO_M2_ARTIFACT_ROOT=/home/wencong/BotColosseo/.worktrees/m3-strong-base \
  PYTHONPATH="$PWD/src" \
  /home/wencong/miniconda3/envs/botcolosseo/bin/python \
  -m pytest tests/integration/test_showcase_smoke.py -q -s
```

Expected: `1 passed`; no ViZDoom worker remains.

- [ ] **Step 5: Document the development and future production commands**

Add `## M4 Showcase foundation` to `script.md`. State explicitly:

- foundation is not an M4 pass;
- M2 PPO/BC validates mechanics only;
- development output is ignored and never updates README;
- production config must not exist until real Strong Base/Aggressive hashes and passing M4 evidence exist.

Document:

```bash
env PYTHONPATH="$PWD/src" \
  /home/wencong/miniconda3/envs/botcolosseo/bin/python \
  scripts/render_showcase.py \
  --config configs/showcase/development.yaml \
  --checkpoint-root /home/wencong/BotColosseo/.worktrees/m3-strong-base \
  --device cuda:0
```

The future command changes only `--config` to `configs/showcase/m4.yaml`.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_showcase_smoke.py tests/unit/test_public_docs.py script.md
git commit -m "test: prove the real showcase rendering path"
```

- [ ] **Step 7: Run full completion audit**

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit -q
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src scripts tests
PYTHONPATH=src /home/wencong/miniconda3/envs/botcolosseo/bin/python \
  -m pytest tests/integration/test_scenario_build.py \
  tests/integration/test_showcase_smoke.py -q
git diff --check
git status --short
git ls-files docs/assets/showcase reports/showcase/m4 artifacts/showcase-development
```

Expected: full unit suite passes with at most the three existing CUDA skips; Ruff/diff pass; scenario build passes; opt-in smoke skips by default; no generated public M4 media or development artifact is tracked.

- [ ] **Step 8: Push and open Draft PR**

Create `/tmp/botcolosseo-m4-showcase-pr.md` with:

```markdown
## What changed

- freezes validation-only development and M4 showcase cases;
- records learner-perspective real ViZDoom episodes from hash-checked policies;
- selects a paired case and highlight deterministically;
- generates MP4/GIF/metrics media with manifest-last publication;
- proves the path with unit tests and a real M2 PPO/BC smoke.

## Why

BotColosseo needs a reproducible GitHub visual layer before the real Aggressive
checkpoint is available. Development media are structurally barred from public
M4 paths, improving delivery speed without making a false style claim.

## Verification

- full unit suite;
- Ruff and diff checks;
- scenario-build integration;
- real CUDA/ViZDoom M2 PPO/BC showcase smoke.

## Scope boundary

This PR does not claim M3 or M4 passed and does not update README with Strong
Base/Aggressive media. Production remains gated on real checkpoint and
evaluation evidence.
```

Then:

```bash
git push origin feat/m4-showcase-foundation
gh pr create \
  --repo kirin-dev/BotColosseo \
  --base main \
  --head feat/m4-showcase-foundation \
  --draft \
  --title "Build the auditable M4 showcase foundation" \
  --body-file /tmp/botcolosseo-m4-showcase-pr.md
gh pr checks --watch --interval 10
```

Expected: Draft PR targets `main`; Portable quality gate succeeds.

---

## Completion Gate

The foundation is complete only when every checkbox above is checked, the real
development smoke has produced non-public MP4/GIF evidence, the full unit suite
and GitHub CI pass, and the Draft PR contains no tracked M4 media or M4-complete
claim. This does not complete M4, Resume-ready, M5, M6, or the overall project.
