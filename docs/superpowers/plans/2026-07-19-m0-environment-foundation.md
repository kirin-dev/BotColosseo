# Bot Colosseo M0 Environment Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a clean public repository and prove that ViZDoom can initialize, expose frames, execute deterministic actions, terminate/reset episodes, and record a short video in headless mode.

**Architecture:** Package all engine setup behind a small `GameSettings`/`create_game` boundary, keep environment diagnostics separate from gameplay, and expose one deterministic smoke runner used by both tests and a CLI. M0 uses ViZDoom's built-in `basic.cfg`; custom Crystal Run assets and gameplay begin only in the separately reviewed M1 plan after this gate passes.

**Tech Stack:** Python 3.10, ViZDoom 1.3.0, PyTorch 2.6, NumPy, ImageIO/FFmpeg, PyYAML, pytest, Ruff, setuptools, Conda.

## Global Constraints

- Use Python `>=3.10,<3.11`; do not rely on the shell's current `python` path.
- Pin ViZDoom to `1.3.0`; record the runtime PyTorch and CUDA versions without requiring CUDA for M0.
- Keep source code under `src/botcolosseo/` and tests under `tests/`.
- Use type annotations on public functions and dataclass fields.
- Disable ViZDoom sound and audio-buffer generation before `game.init()` in headless mode.
- Always close `DoomGame` in `finally`, including initialization and recording failures.
- Do not commit the personal resume PDF, `_vizdoom.ini`, `_vizdoom/`, checkpoints, runtime logs, or generated videos.
- Do not commit original Doom/Doom II commercial assets; document ViZDoom and Freedoom attribution.
- M0 passes only when unit tests, Ruff, the real headless integration test, and the required-video smoke command all exit with code 0.
- Do not start custom maps, PPO, multiplayer, style shaping, or league work in this plan.

## Plan Boundaries

The approved design contains several gate-dependent subsystems. They must be planned separately in this order:

1. This document: repository hygiene and M0 environment reliability.
2. After M0 passes: M1 Crystal Run UDMF/ACS scenario, event protocol, single-agent subtasks, and scripted Teachers.
3. After M1 passes: M2 synchronous duel environment, demonstrations, BC, and recurrent PPO.
4. After M2 passes: M3 historical pool, PFSP, and Strong Base evaluation.
5. After M3 passes: M4 Aggressive vertical slice and skill-retention ablations.
6. After M4 passes: M5 Defensive, Explorer, and difficulty controller.
7. After M5 passes: M6 user study, artifacts, and public release.

This split prevents later interfaces from being invented before engine and scenario evidence exists. `Plan.md` remains the authoritative cross-milestone specification.

## Target File Map

| Path | Responsibility |
|---|---|
| `.gitignore` | Exclude personal, generated, runtime, and large training files |
| `LICENSE` | MIT license for Bot Colosseo source code |
| `THIRD_PARTY_NOTICES.md` | ViZDoom/Freedoom attribution and commercial-asset prohibition |
| `pyproject.toml` | Authoritative package metadata, dependencies, pytest and Ruff configuration |
| `env.yml` | Reproducible Conda environment that installs the local package |
| `requirements.txt` | Thin pip compatibility entrypoint delegating to `pyproject.toml` |
| `src/botcolosseo/__init__.py` | Package version |
| `src/botcolosseo/runtime.py` | Runtime and accelerator diagnostics |
| `src/botcolosseo/envs/vizdoom_game.py` | ViZDoom construction and initialization boundary |
| `src/botcolosseo/envs/video.py` | Frame normalization and atomic MP4 writing |
| `src/botcolosseo/envs/smoke.py` | Deterministic M0 episode runner and structured summary |
| `src/botcolosseo/cli/check_env.py` | JSON runtime-report CLI |
| `src/botcolosseo/cli/smoke.py` | Headless smoke-test CLI |
| `scripts/check_env.py` | Repository-level runtime command entrypoint |
| `scripts/smoke_vizdoom.py` | Repository-level smoke command entrypoint |
| `tests/unit/` | Pure unit tests with fake engine/writer boundaries |
| `tests/integration/` | Real ViZDoom and FFmpeg tests |
| `docs/milestones/m0.md` | M0 commands, gates, expected artifacts, and troubleshooting |
| `README.md` | Truthful current public status and quick-start commands |

---

### Task 1: Establish Public-Repository Hygiene and Licensing

**Files:**
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `THIRD_PARTY_NOTICES.md`
- Commit: `Plan.md`
- Commit: `script.md`
- Commit: `docs/superpowers/plans/2026-07-19-m0-environment-foundation.md`

**Interfaces:**
- Consumes: the initialized Git repository on branch `main`.
- Produces: a clean public tracking boundary; subsequent tasks may safely use explicit `git add` commands.

- [ ] **Step 1: Verify the current repository leaks generated and personal files into status**

Run:

```bash
git check-ignore -q _vizdoom.ini
```

Expected: exit code `1`, proving `_vizdoom.ini` is not ignored yet.

Run:

```bash
git check-ignore -q '盛文聪_中国科学院自动化研究所_强化学习.pdf'
```

Expected: exit code `1`, proving the personal resume is not ignored yet.

- [ ] **Step 2: Create the repository ignore boundary**

Create `.gitignore` with exactly:

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/
build/
dist/
.venv/

# Git worktrees
.worktrees/

# ViZDoom runtime state
/_vizdoom/
/_vizdoom.ini
*.lmp

# Experiments and generated media
/runs/
/checkpoints/
/artifacts/
/videos/
/wandb/
/tensorboard/

# Local IDE and OS files
.idea/
.vscode/
.DS_Store

# Personal application material
/盛文聪_中国科学院自动化研究所_强化学习.pdf
```

- [ ] **Step 3: Verify the ignore rules**

Run:

```bash
git check-ignore -v _vizdoom.ini '盛文聪_中国科学院自动化研究所_强化学习.pdf'
```

Expected: two lines pointing to `.gitignore`, one for each file.

Run:

```bash
git status --short
```

Expected: neither `_vizdoom.ini` nor the resume PDF appears.

- [ ] **Step 4: Add the project source-code license**

Create `LICENSE` with exactly:

```text
MIT License

Copyright (c) 2026 Wencong Sheng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 5: Add third-party notices**

Create `THIRD_PARTY_NOTICES.md` with exactly:

```markdown
# Third-Party Notices

Bot Colosseo source code is licensed under the MIT License. Third-party
software and game data retain their own licenses.

## ViZDoom

ViZDoom is developed by the Farama Foundation and contributors. Code original
to ViZDoom is distributed under the MIT License; the bundled ZDoom-derived
engine includes components under additional compatible licenses. See the
ViZDoom distribution and https://github.com/Farama-Foundation/ViZDoom for the
complete notices.

## Freedoom

Freedoom game data is Copyright 2001-2026 Contributors to the Freedoom project
and is distributed under the BSD 3-Clause License. See
https://freedoom.github.io/ and the license shipped with the selected Freedoom
release.

Bot Colosseo does not redistribute assets from the commercial Doom or Doom II
games. Contributors must not add commercial IWADs, textures, sounds, maps, or
other proprietary game data to this repository.
```

- [ ] **Step 6: Commit the approved specification and repository boundary**

Run:

```bash
git add .gitignore LICENSE THIRD_PARTY_NOTICES.md Plan.md script.md docs/superpowers/plans/2026-07-19-m0-environment-foundation.md
git diff --cached --check
git commit -m "docs: define Bot Colosseo technical direction"
```

Expected: `git diff --cached --check` exits `0`; commit creates the first repository commit without the resume or runtime files.

---

### Task 2: Create the Installable Python Package

**Files:**
- Create: `pyproject.toml`
- Modify: `env.yml`
- Modify: `requirements.txt`
- Create: `README.md`
- Create: `src/botcolosseo/__init__.py`
- Create: `tests/unit/test_package.py`

**Interfaces:**
- Consumes: Python 3.10 and the repository root.
- Produces: importable package `botcolosseo` with `botcolosseo.__version__ == "0.1.0"`.

- [ ] **Step 1: Write the failing package test**

Create `tests/unit/test_package.py`:

```python
import botcolosseo


def test_package_version() -> None:
    assert botcolosseo.__version__ == "0.1.0"
```

- [ ] **Step 2: Run the test to verify the package is absent**

Run:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_package.py -v
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'botcolosseo'`.

- [ ] **Step 3: Create authoritative package metadata**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "botcolosseo"
version = "0.1.0"
description = "Skill-preserving policy shaping for controllable visual game bots"
readme = "README.md"
requires-python = ">=3.10,<3.11"
license = { file = "LICENSE" }
authors = [{ name = "Wencong Sheng" }]
dependencies = [
  "vizdoom==1.3.0",
  "gymnasium>=1.0,<2.0",
  "numpy>=1.24,<3.0",
  "PyYAML>=6.0,<7.0",
  "imageio>=2.34,<3.0",
  "imageio-ffmpeg>=0.5,<1.0",
]

[project.optional-dependencies]
training = [
  "torch>=2.6,<2.7",
  "pandas>=2.0,<3.0",
  "matplotlib>=3.7,<4.0",
  "opencv-python-headless>=4.8,<6.0",
  "tensorboard>=2.15,<3.0",
  "tqdm>=4.66,<5.0",
  "psutil>=5.9,<8.0",
  "scipy>=1.11,<2.0",
  "seaborn>=0.13,<1.0",
]
dev = [
  "pytest>=8.0,<10.0",
  "pytest-timeout>=2.3,<3.0",
  "pytest-cov>=5.0,<8.0",
  "ruff>=0.9,<1.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
markers = [
  "integration: starts real ViZDoom or FFmpeg processes",
]

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

Replace `env.yml` with:

```yaml
name: botcolosseo

channels:
  - pytorch
  - nvidia
  - conda-forge

dependencies:
  - python=3.10
  - pytorch=2.6
  - pytorch-cuda=12.4
  - pip
  - pip:
      - -e .[training,dev]
```

Replace `requirements.txt` with:

```text
-e .[training,dev]
```

Create `src/botcolosseo/__init__.py`:

```python
"""Bot Colosseo package."""

__version__ = "0.1.0"
```

Create `README.md` with a truthful pre-M0 status so package metadata can resolve it:

```markdown
# Bot Colosseo

Goal-oriented controllable visual game bots via skill-preserving policy shaping.

The technical design is approved in [Plan.md](Plan.md). Environment reliability
is currently being established; no custom scenario, trained policy, or style
result is claimed yet.
```

- [ ] **Step 4: Install the package in editable mode**

Run:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pip install -e '.[training,dev]'
```

Expected: exit code `0` and an editable `botcolosseo==0.1.0` installation.

- [ ] **Step 5: Run the package test and linter**

Run:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_package.py -v
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src tests
```

Expected: one passing test and Ruff exit code `0`.

- [ ] **Step 6: Commit the package foundation**

Run:

```bash
git add pyproject.toml env.yml requirements.txt README.md src/botcolosseo/__init__.py tests/unit/test_package.py
git diff --cached --check
git commit -m "build: add installable Python package"
```

Expected: commit succeeds and `git status --short` shows only the obsolete untracked `test.py`.

---

### Task 3: Add Structured Runtime Diagnostics

**Files:**
- Create: `src/botcolosseo/runtime.py`
- Create: `src/botcolosseo/cli/__init__.py`
- Create: `src/botcolosseo/cli/check_env.py`
- Create: `scripts/check_env.py`
- Create: `tests/unit/test_runtime.py`

**Interfaces:**
- Consumes: installed `torch` and `vizdoom` packages.
- Produces: `RuntimeReport`, `inspect_runtime() -> RuntimeReport`, and a JSON CLI that distinguishes CUDA availability from M0 validity.

- [ ] **Step 1: Write the failing runtime-report test**

Create `tests/unit/test_runtime.py`:

```python
import json

from botcolosseo.runtime import RuntimeReport


def test_runtime_report_serializes_stable_keys() -> None:
    report = RuntimeReport(
        python_version="3.10.20",
        python_executable="/tmp/python",
        torch_version="2.6.0+cu124",
        vizdoom_version="1.3.0",
        cuda_available=False,
        cuda_version="12.4",
        gpu_names=(),
    )

    payload = json.loads(report.to_json())

    assert payload == {
        "cuda_available": False,
        "cuda_version": "12.4",
        "gpu_names": [],
        "python_executable": "/tmp/python",
        "python_version": "3.10.20",
        "torch_version": "2.6.0+cu124",
        "vizdoom_version": "1.3.0",
    }
```

- [ ] **Step 2: Run the test to verify the interface is absent**

Run:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_runtime.py -v
```

Expected: FAIL during collection because `botcolosseo.runtime` does not exist.

- [ ] **Step 3: Implement the runtime report**

Create `src/botcolosseo/runtime.py`:

```python
from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass

import torch
import vizdoom as vzd


@dataclass(frozen=True)
class RuntimeReport:
    python_version: str
    python_executable: str
    torch_version: str
    vizdoom_version: str
    cuda_available: bool
    cuda_version: str | None
    gpu_names: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def inspect_runtime() -> RuntimeReport:
    cuda_available = torch.cuda.is_available()
    gpu_names = (
        tuple(torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count()))
        if cuda_available
        else ()
    )
    return RuntimeReport(
        python_version=platform.python_version(),
        python_executable=sys.executable,
        torch_version=torch.__version__,
        vizdoom_version=vzd.__version__,
        cuda_available=cuda_available,
        cuda_version=torch.version.cuda,
        gpu_names=gpu_names,
    )
```

Create an empty `src/botcolosseo/cli/__init__.py`.

Create `src/botcolosseo/cli/check_env.py`:

```python
from __future__ import annotations

import sys

from botcolosseo.runtime import inspect_runtime


def main() -> int:
    report = inspect_runtime()
    print(report.to_json())
    python_ok = sys.version_info[:2] == (3, 10)
    vizdoom_ok = report.vizdoom_version == "1.3.0"
    return 0 if python_ok and vizdoom_ok else 1
```

Create `scripts/check_env.py`:

```python
from botcolosseo.cli.check_env import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run unit and real diagnostic checks**

Run:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_runtime.py -v
/home/wencong/miniconda3/envs/botcolosseo/bin/python scripts/check_env.py
```

Expected: unit test passes; the CLI exits `0`, reports Python 3.10 and ViZDoom 1.3.0, and may truthfully report `cuda_available: false` without failing M0.

- [ ] **Step 5: Commit runtime diagnostics**

Run:

```bash
git add src/botcolosseo/runtime.py src/botcolosseo/cli scripts/check_env.py tests/unit/test_runtime.py
git diff --cached --check
git commit -m "feat: add runtime diagnostics"
```

Expected: commit succeeds.

---

### Task 4: Encapsulate ViZDoom Initialization

**Files:**
- Create: `src/botcolosseo/envs/__init__.py`
- Create: `src/botcolosseo/envs/vizdoom_game.py`
- Create: `tests/unit/test_vizdoom_game.py`

**Interfaces:**
- Consumes: `GameSettings(config_path: Path, seed: int, visible: bool, screen_format: vzd.ScreenFormat)`.
- Produces: `create_game(settings, game_factory=vzd.DoomGame) -> vzd.DoomGame`, returning an initialized, silent engine that the caller must close.

- [ ] **Step 1: Write the failing game-factory test**

Create `tests/unit/test_vizdoom_game.py`:

```python
from pathlib import Path

import vizdoom as vzd

from botcolosseo.envs.vizdoom_game import GameSettings, create_game


class FakeGame:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.closed = False

    def load_config(self, path: str) -> bool:
        self.calls.append(("load_config", path))
        return True

    def set_seed(self, value: int) -> None:
        self.calls.append(("set_seed", value))

    def set_window_visible(self, value: bool) -> None:
        self.calls.append(("set_window_visible", value))

    def set_sound_enabled(self, value: bool) -> None:
        self.calls.append(("set_sound_enabled", value))

    def set_audio_buffer_enabled(self, value: bool) -> None:
        self.calls.append(("set_audio_buffer_enabled", value))

    def set_screen_format(self, value: vzd.ScreenFormat) -> None:
        self.calls.append(("set_screen_format", value))

    def set_mode(self, value: vzd.Mode) -> None:
        self.calls.append(("set_mode", value))

    def init(self) -> None:
        self.calls.append(("init", None))

    def close(self) -> None:
        self.closed = True


def test_create_game_applies_headless_silent_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "basic.cfg"
    config_path.write_text("doom_map = map01\n", encoding="utf-8")
    fake = FakeGame()
    settings = GameSettings(config_path=config_path, seed=7)

    result = create_game(settings, game_factory=lambda: fake)

    assert result is fake
    assert ("set_seed", 7) in fake.calls
    assert ("set_window_visible", False) in fake.calls
    assert ("set_sound_enabled", False) in fake.calls
    assert ("set_audio_buffer_enabled", False) in fake.calls
    assert fake.calls[-1] == ("init", None)
    assert not fake.closed
```

- [ ] **Step 2: Run the test to verify the interface is absent**

Run:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_vizdoom_game.py -v
```

Expected: FAIL during collection because `botcolosseo.envs.vizdoom_game` does not exist.

- [ ] **Step 3: Implement the game factory**

Create an empty `src/botcolosseo/envs/__init__.py`.

Create `src/botcolosseo/envs/vizdoom_game.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import vizdoom as vzd


@dataclass(frozen=True)
class GameSettings:
    config_path: Path
    seed: int = 0
    visible: bool = False
    screen_format: vzd.ScreenFormat = vzd.ScreenFormat.GRAY8


def create_game(
    settings: GameSettings,
    game_factory: Callable[[], vzd.DoomGame] = vzd.DoomGame,
) -> vzd.DoomGame:
    config_path = settings.config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"ViZDoom config does not exist: {config_path}")

    game = game_factory()
    try:
        if not game.load_config(str(config_path)):
            raise ValueError(f"ViZDoom rejected config: {config_path}")
        game.set_seed(settings.seed)
        game.set_window_visible(settings.visible)
        game.set_sound_enabled(False)
        game.set_audio_buffer_enabled(False)
        game.set_screen_format(settings.screen_format)
        game.set_mode(vzd.Mode.PLAYER)
        game.init()
    except BaseException:
        game.close()
        raise
    return game
```

- [ ] **Step 4: Run unit tests and Ruff**

Run:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_vizdoom_game.py -v
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src tests
```

Expected: game-factory test passes and Ruff exits `0`.

- [ ] **Step 5: Commit the engine boundary**

Run:

```bash
git add src/botcolosseo/envs tests/unit/test_vizdoom_game.py
git diff --cached --check
git commit -m "feat: add silent ViZDoom game factory"
```

Expected: commit succeeds.

---

### Task 5: Add Atomic MP4 Recording

**Files:**
- Create: `src/botcolosseo/envs/video.py`
- Create: `tests/unit/test_video.py`

**Interfaces:**
- Consumes: grayscale, HWC RGB, or CHW RGB NumPy frames.
- Produces: `normalize_rgb_frame(frame) -> np.ndarray` and `write_mp4(frames, output_path, fps) -> Path`; incomplete temporary files are removed on failure.

- [ ] **Step 1: Write failing frame-normalization tests**

Create `tests/unit/test_video.py`:

```python
from pathlib import Path

import numpy as np

from botcolosseo.envs.video import normalize_rgb_frame, write_mp4


def test_normalize_rgb_frame_accepts_grayscale() -> None:
    frame = np.zeros((84, 84), dtype=np.uint8)

    result = normalize_rgb_frame(frame)

    assert result.shape == (84, 84, 3)
    assert result.dtype == np.uint8


def test_normalize_rgb_frame_accepts_chw() -> None:
    frame = np.zeros((3, 84, 84), dtype=np.uint8)

    result = normalize_rgb_frame(frame)

    assert result.shape == (84, 84, 3)


def test_write_mp4_uses_atomic_target(tmp_path: Path, monkeypatch) -> None:
    appended: list[np.ndarray] = []

    class FakeWriter:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def append_data(self, frame: np.ndarray) -> None:
            appended.append(frame)

    def fake_get_writer(path: Path, **kwargs):
        Path(path).touch()
        return FakeWriter()

    monkeypatch.setattr("botcolosseo.envs.video.imageio.get_writer", fake_get_writer)
    output = tmp_path / "smoke.mp4"

    result = write_mp4([np.zeros((8, 8), dtype=np.uint8)], output, fps=10)

    assert result == output
    assert output.is_file()
    assert len(appended) == 1
```

- [ ] **Step 2: Run the tests to verify the module is absent**

Run:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_video.py -v
```

Expected: FAIL during collection because `botcolosseo.envs.video` does not exist.

- [ ] **Step 3: Implement frame normalization and atomic video writing**

Create `src/botcolosseo/envs/video.py`:

```python
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from numpy.typing import NDArray


def normalize_rgb_frame(frame: NDArray[np.generic]) -> NDArray[np.uint8]:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    elif array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unsupported frame shape: {array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]
    return np.ascontiguousarray(array, dtype=np.uint8)


def write_mp4(
    frames: Iterable[NDArray[np.generic]],
    output_path: Path,
    fps: int,
) -> Path:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    try:
        with imageio.get_writer(
            temporary_path,
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libx264",
            macro_block_size=None,
        ) as writer:
            frame_count = 0
            for frame in frames:
                writer.append_data(normalize_rgb_frame(frame))
                frame_count += 1
        if frame_count == 0:
            raise ValueError("Cannot write an empty video")
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path
```

- [ ] **Step 4: Run tests and Ruff**

Run:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_video.py -v
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src tests
```

Expected: three video unit tests pass and Ruff exits `0`.

- [ ] **Step 5: Commit the video boundary**

Run:

```bash
git add src/botcolosseo/envs/video.py tests/unit/test_video.py
git diff --cached --check
git commit -m "feat: add atomic smoke video writer"
```

Expected: commit succeeds.

---

### Task 6: Implement the Deterministic Smoke Runner

**Files:**
- Create: `src/botcolosseo/envs/smoke.py`
- Create: `tests/unit/test_smoke.py`

**Interfaces:**
- Consumes: `GameSettings`, `episodes`, `max_decisions`, `frame_skip`, optional MP4 path, and `require_video`.
- Produces: `EpisodeSummary`, `SmokeSummary`, and `run_smoke(...) -> SmokeSummary`; game closure is guaranteed.

- [ ] **Step 1: Write failing smoke-runner tests**

Create `tests/unit/test_smoke.py`:

```python
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from botcolosseo.envs.smoke import run_smoke
from botcolosseo.envs.vizdoom_game import GameSettings


class FakeGame:
    def __init__(self) -> None:
        self.tic = 0
        self.closed = False

    def new_episode(self) -> None:
        self.tic = 0

    def is_episode_finished(self) -> bool:
        return self.tic >= 3

    def get_state(self):
        return SimpleNamespace(screen_buffer=np.zeros((84, 84), dtype=np.uint8))

    def get_available_buttons_size(self) -> int:
        return 2

    def make_action(self, action: list[float], frame_skip: int) -> float:
        assert sum(action) == 1.0
        self.tic += 1
        return 1.0

    def get_total_reward(self) -> float:
        return 3.0

    def close(self) -> None:
        self.closed = True


def test_run_smoke_terminates_and_closes_game(tmp_path: Path) -> None:
    fake = FakeGame()
    settings = GameSettings(config_path=tmp_path / "unused.cfg")

    summary = run_smoke(
        settings,
        episodes=2,
        max_decisions=5,
        frame_skip=4,
        game_builder=lambda unused: fake,
    )

    assert summary.all_terminated
    assert [episode.decisions for episode in summary.episodes] == [3, 3]
    assert fake.closed


def test_optional_video_failure_does_not_fail_smoke(tmp_path: Path) -> None:
    fake = FakeGame()
    settings = GameSettings(config_path=tmp_path / "unused.cfg")

    def failing_writer(frames, path, fps):
        raise RuntimeError("encoder unavailable")

    summary = run_smoke(
        settings,
        episodes=1,
        max_decisions=5,
        frame_skip=4,
        video_path=tmp_path / "smoke.mp4",
        require_video=False,
        game_builder=lambda unused: fake,
        video_writer=failing_writer,
    )

    assert summary.all_terminated
    assert summary.video_error == "encoder unavailable"
    assert fake.closed
```

- [ ] **Step 2: Run tests to verify the interface is absent**

Run:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_smoke.py -v
```

Expected: FAIL during collection because `botcolosseo.envs.smoke` does not exist.

- [ ] **Step 3: Implement the smoke runner**

Create `src/botcolosseo/envs/smoke.py`:

```python
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from botcolosseo.envs.video import write_mp4
from botcolosseo.envs.vizdoom_game import GameSettings, create_game


@dataclass(frozen=True)
class EpisodeSummary:
    index: int
    decisions: int
    total_reward: float
    terminated: bool
    first_frame_shape: tuple[int, ...]


@dataclass(frozen=True)
class SmokeSummary:
    episodes: tuple[EpisodeSummary, ...]
    all_terminated: bool
    video_path: str | None
    video_error: str | None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _one_hot_action(button_count: int, decision: int) -> list[float]:
    if button_count <= 0:
        raise RuntimeError("ViZDoom scenario exposes no available buttons")
    action = [0.0] * button_count
    action[decision % button_count] = 1.0
    return action


def run_smoke(
    settings: GameSettings,
    *,
    episodes: int = 2,
    max_decisions: int = 100,
    frame_skip: int = 4,
    video_path: Path | None = None,
    video_fps: int = 10,
    require_video: bool = False,
    game_builder: Callable[[GameSettings], Any] = create_game,
    video_writer: Callable[[list[np.ndarray], Path, int], Path] = write_mp4,
) -> SmokeSummary:
    if episodes <= 0 or max_decisions <= 0 or frame_skip <= 0:
        raise ValueError("episodes, max_decisions, and frame_skip must be positive")
    if require_video and video_path is None:
        raise ValueError("require_video=True requires video_path")

    game = game_builder(settings)
    episode_summaries: list[EpisodeSummary] = []
    recorded_frames: list[np.ndarray] = []
    video_error: str | None = None
    try:
        for episode_index in range(episodes):
            game.new_episode()
            decisions = 0
            first_frame_shape: tuple[int, ...] = ()
            while not game.is_episode_finished() and decisions < max_decisions:
                state = game.get_state()
                if state is None:
                    raise RuntimeError("ViZDoom returned no state before termination")
                frame = np.asarray(state.screen_buffer)
                if not first_frame_shape:
                    first_frame_shape = tuple(frame.shape)
                if video_path is not None and episode_index == 0:
                    recorded_frames.append(frame.copy())
                action = _one_hot_action(game.get_available_buttons_size(), decisions)
                game.make_action(action, frame_skip)
                decisions += 1
            terminated = game.is_episode_finished()
            episode_summaries.append(
                EpisodeSummary(
                    index=episode_index,
                    decisions=decisions,
                    total_reward=float(game.get_total_reward()),
                    terminated=terminated,
                    first_frame_shape=first_frame_shape,
                )
            )

        if video_path is not None:
            try:
                video_writer(recorded_frames, video_path, video_fps)
            except Exception as exc:
                video_error = str(exc)
                if require_video:
                    raise RuntimeError(f"Required smoke video failed: {exc}") from exc
    finally:
        game.close()

    return SmokeSummary(
        episodes=tuple(episode_summaries),
        all_terminated=all(item.terminated for item in episode_summaries),
        video_path=str(video_path.resolve()) if video_path is not None and video_error is None else None,
        video_error=video_error,
    )
```

- [ ] **Step 4: Run tests and Ruff**

Run:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_smoke.py -v
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src tests
```

Expected: two smoke-runner tests pass and Ruff exits `0`.

- [ ] **Step 5: Commit the deterministic runner**

Run:

```bash
git add src/botcolosseo/envs/smoke.py tests/unit/test_smoke.py
git diff --cached --check
git commit -m "feat: add deterministic ViZDoom smoke runner"
```

Expected: commit succeeds.

---

### Task 7: Add the Smoke CLI and Real Integration Gates

**Files:**
- Create: `src/botcolosseo/cli/smoke.py`
- Create: `scripts/smoke_vizdoom.py`
- Create: `tests/integration/test_vizdoom_smoke.py`
- Delete: `test.py`

**Interfaces:**
- Consumes: ViZDoom built-in `basic.cfg` by default, with optional explicit config and recording path.
- Produces: JSON smoke summary on stdout and process exit `0` only when all requested episodes terminate and required recording succeeds.

- [ ] **Step 1: Write the real integration tests**

Create `tests/integration/test_vizdoom_smoke.py`:

```python
import subprocess
import sys
from pathlib import Path

import pytest
import vizdoom as vzd

from botcolosseo.envs.smoke import run_smoke
from botcolosseo.envs.vizdoom_game import GameSettings


@pytest.mark.integration
def test_basic_scenario_terminates_and_resets() -> None:
    settings = GameSettings(config_path=Path(vzd.scenarios_path) / "basic.cfg", seed=17)

    summary = run_smoke(settings, episodes=2, max_decisions=100, frame_skip=4)

    assert summary.all_terminated
    assert len(summary.episodes) == 2
    assert all(item.first_frame_shape for item in summary.episodes)


@pytest.mark.integration
def test_basic_scenario_records_mp4(tmp_path: Path) -> None:
    settings = GameSettings(config_path=Path(vzd.scenarios_path) / "basic.cfg", seed=17)
    video_path = tmp_path / "smoke.mp4"

    summary = run_smoke(
        settings,
        episodes=1,
        max_decisions=100,
        frame_skip=4,
        video_path=video_path,
        require_video=True,
    )

    assert summary.all_terminated
    assert summary.video_error is None
    assert video_path.stat().st_size > 0


@pytest.mark.integration
def test_smoke_cli_returns_success() -> None:
    repository_root = Path(__file__).parents[2]

    result = subprocess.run(
        [sys.executable, "scripts/smoke_vizdoom.py", "--episodes", "1"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"all_terminated": true' in result.stdout
```

- [ ] **Step 2: Run the integration tests against the current engine path**

Run only the CLI test:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/integration/test_vizdoom_smoke.py::test_smoke_cli_returns_success -v
```

Expected: FAIL because `scripts/smoke_vizdoom.py` does not exist. The subprocess returns a non-zero code and the assertion reports the missing script.

- [ ] **Step 3: Implement the CLI**

Create `src/botcolosseo/cli/smoke.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import vizdoom as vzd

from botcolosseo.envs.smoke import run_smoke
from botcolosseo.envs.vizdoom_game import GameSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Bot Colosseo ViZDoom smoke gate")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(vzd.scenarios_path) / "basic.cfg",
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-decisions", type=int, default=100)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--require-video", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_smoke(
        GameSettings(config_path=args.config, seed=args.seed),
        episodes=args.episodes,
        max_decisions=args.max_decisions,
        frame_skip=args.frame_skip,
        video_path=args.record,
        require_video=args.require_video,
    )
    print(summary.to_json())
    if not summary.all_terminated:
        return 1
    if args.require_video and summary.video_error is not None:
        return 1
    return 0
```

Create `scripts/smoke_vizdoom.py`:

```python
from botcolosseo.cli.smoke import main


if __name__ == "__main__":
    raise SystemExit(main())
```

Delete the obsolete root-level `test.py`; its functionality is replaced by the tested package API and CLI.

- [ ] **Step 4: Run the full M0 functional commands**

Run:

```bash
timeout 30s /home/wencong/miniconda3/envs/botcolosseo/bin/python scripts/smoke_vizdoom.py
timeout 30s /home/wencong/miniconda3/envs/botcolosseo/bin/python scripts/smoke_vizdoom.py --episodes 1 --record videos/m0-smoke.mp4 --require-video
```

Expected: both commands exit `0`; JSON reports `all_terminated: true`; the second reports a non-null `video_path` and creates a non-empty `videos/m0-smoke.mp4`.

- [ ] **Step 5: Run all tests and Ruff**

Run:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest -v
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src tests scripts
```

Expected: all unit and integration tests pass; Ruff exits `0`.

- [ ] **Step 6: Commit the public smoke interface**

Run:

```bash
git add src/botcolosseo/cli/smoke.py scripts/smoke_vizdoom.py tests/integration/test_vizdoom_smoke.py
git diff --cached --check
git commit -m "test: add real ViZDoom smoke gates"
```

Expected: commit succeeds and the obsolete `test.py` is removed.

---

### Task 8: Document and Verify the M0 Gate

**Files:**
- Create: `docs/milestones/m0.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the verified commands and artifacts from Tasks 1–7.
- Produces: a truthful public quick start and an explicit M0 acceptance checklist; no M1 claim is made.

- [ ] **Step 1: Create the M0 runbook**

Create `docs/milestones/m0.md`:

```markdown
# Milestone 0: ViZDoom Environment Reliability

M0 proves engine reliability only. It does not include the Crystal Run map,
multiplayer, learning, or style shaping.

## Environment

```bash
conda env create -f env.yml
conda activate botcolosseo
python scripts/check_env.py
```

CUDA availability is reported but is not required for M0.

## Required gates

```bash
python -m pytest -v
python -m ruff check src tests scripts
timeout 30s python scripts/smoke_vizdoom.py
timeout 30s python scripts/smoke_vizdoom.py \
  --episodes 1 \
  --record videos/m0-smoke.mp4 \
  --require-video
```

M0 passes only if every command exits with code 0, both smoke episodes
terminate naturally, reset produces a new valid frame, and the MP4 is non-empty.

## Common failures

- A shell resolving to Python 3.7: activate the `botcolosseo` environment or
  use its absolute interpreter path.
- PulseAudio `pa_write()` errors: confirm that both `set_sound_enabled(False)`
  and `set_audio_buffer_enabled(False)` execute before `game.init()`.
- CUDA/NVML unavailable: record it in `check_env.py`; debug it before training,
  but do not treat it as an M0 engine failure.
- A hung engine process: fail the 30-second gate and inspect process cleanup;
  do not continue to M1.
```

- [ ] **Step 2: Create a truthful public README for the completed gate**

Create `README.md`:

```markdown
# Bot Colosseo

Goal-oriented controllable visual game bots via skill-preserving policy shaping.

Bot Colosseo studies how to train a strong visual game Bot and then shape it
into player-recognizable Aggressive, Defensive, and Explorer styles without
discarding its task skill. The approved technical design is in [Plan.md](Plan.md).

## Current status

Milestone 0 verifies reproducible headless ViZDoom initialization, deterministic
actions, episode termination/reset, and MP4 recording. The Crystal Run scenario,
learning pipeline, multiplayer, and style models are planned but are not claimed
as implemented.

## Quick start

```bash
conda env create -f env.yml
conda activate botcolosseo
python scripts/check_env.py
python -m pytest -v
python scripts/smoke_vizdoom.py \
  --episodes 1 \
  --record videos/m0-smoke.mp4 \
  --require-video
```

See [the M0 runbook](docs/milestones/m0.md) for acceptance criteria and
troubleshooting.

## Licensing

Bot Colosseo source code is MIT licensed. ViZDoom and Freedoom retain their own
licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This repository
does not distribute commercial Doom assets.
```

- [ ] **Step 3: Run the complete fresh verification suite**

Run:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python scripts/check_env.py
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest -v
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src tests scripts
timeout 30s /home/wencong/miniconda3/envs/botcolosseo/bin/python scripts/smoke_vizdoom.py
timeout 30s /home/wencong/miniconda3/envs/botcolosseo/bin/python scripts/smoke_vizdoom.py --episodes 1 --record videos/m0-smoke.mp4 --require-video
test -s videos/m0-smoke.mp4
git diff --check
```

Expected: every command exits `0`; pytest reports zero failures; Ruff reports no errors; both smoke summaries report `all_terminated: true`; the video-size test succeeds; `git diff --check` emits nothing.

- [ ] **Step 4: Commit M0 documentation**

Run:

```bash
git add README.md docs/milestones/m0.md
git diff --cached --check
git commit -m "docs: record Milestone 0 gates"
```

Expected: commit succeeds.

- [ ] **Step 5: Confirm the milestone boundary**

Run:

```bash
git status --short
git log --oneline --decorate -8
```

Expected: the worktree is clean, generated video and personal files remain ignored, and the log contains the M0 commits from this plan. Stop here and request review before writing or executing the M1 plan.

## Plan Self-Review Record

- Spec coverage: repository hygiene, correct interpreter reporting, silent headless initialization, frame access, deterministic action execution, termination/reset, recording, cleanup, tests, licensing, truthful documentation, and the M0 gate are each assigned to a task.
- Scope exclusions: custom scenarios, ACS, multiplayer, RL, styles, difficulty, and user study remain behind their approved gates.
- Type consistency: `GameSettings`, `create_game`, `RuntimeReport`, `EpisodeSummary`, `SmokeSummary`, `run_smoke`, `normalize_rgb_frame`, and `write_mp4` use the same names and signatures across producers and consumers.
- No command in this plan has been executed as part of writing the plan.
