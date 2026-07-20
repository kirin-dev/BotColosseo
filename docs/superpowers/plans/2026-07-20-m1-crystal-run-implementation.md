# Bot Colosseo M1 Crystal Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a reproducible single-instance Crystal Run Arena with typed observations, an auditable ACS event protocol, deterministic task splits, five scripted Teachers, and quantitative M1 evidence.

**Architecture:** Reviewable UDMF/ACS/YAML sources are compiled with ACC and packed by a small tested Python WAD writer into one tracked scenario artifact. A typed `SingleAgentTaskEnv` separates legal Actor observations from privileged Teacher state, decodes monotonic ACS counters into immutable events, and exposes the same 13 macro actions that later learning stages will use. Deterministic Teachers and frozen manifests drive a machine-verifiable 500-episode capability gate.

**Tech Stack:** Python 3.10, ViZDoom 1.3.0, ACC 1.60, UDMF, ACS, NumPy, PyYAML, ImageIO/FFmpeg, pytest, Ruff, setuptools.

## Global Constraints

- Preserve `Plan.md` as the cross-milestone authority and `docs/superpowers/specs/2026-07-20-m1-crystal-run-design.md` as the M1 authority.
- Use `/home/wencong/miniconda3/envs/botcolosseo/bin/python` for every Python command.
- Resolve ACC from `--acc`, `ACC_PATH`, or `PATH`; never assume `/home/wencong/.local/bin/acc` in package code.
- Keep the tracked WAD runnable without ACC; ACC is required only to rebuild it.
- Use only Freedoom-compatible built-in assets and self-authored map/script data.
- Keep Actor input, privileged state, and event data in distinct frozen types.
- Actor observations must not contain positions, angles, region IDs, depth, labels, automap, target coordinates, or invisible-object state.
- `USER1`–`USER20` follow protocol v1 exactly; global ACS variable 0 remains reserved for ViZDoom reward.
- Preserve the 13 macro-action IDs from `Plan.md`; expand them only after a separately reviewed route change.
- Train, validation, and test seeds must be pairwise disjoint and reproducible from a master seed.
- M1 uses no `nohup` job and performs no learned-policy training.
- A real-engine command always has a 30-second per-process timeout or pytest timeout.
- M1 passes only after unit/integration tests, Ruff, deterministic rebuild, event negative tests, five 100-seed gates, evidence packaging, and required video all pass.
- Make one focused commit per task; do not commit transient build directories, development logs, or videos.

## Target File Map

| Path | Responsibility |
|---|---|
| `src/botcolosseo/scenarios/wad.py` | Minimal deterministic PWAD packing and inspection |
| `src/botcolosseo/scenarios/build.py` | ACC discovery, ACS compilation, WAD/manifest build |
| `src/botcolosseo/scenarios/regions.py` | Region schema, membership, and graph routing |
| `src/botcolosseo/scenarios/splits.py` | Frozen disjoint task-case manifests |
| `assets/scenarios/crystal_run/src/map.udmf` | Single arena geometry source |
| `assets/scenarios/crystal_run/src/crystal_run.acs` | Task setup and protocol-v1 counters |
| `assets/scenarios/crystal_run/src/regions.yaml` | Stable regions, waypoints, and adjacency |
| `assets/scenarios/crystal_run/src/task_variants.yaml` | Task marker and timeout definitions |
| `assets/scenarios/crystal_run/crystal_run.cfg` | ViZDoom runtime configuration |
| `assets/scenarios/crystal_run/crystal_run.wad` | Tracked runnable scenario artifact |
| `assets/scenarios/crystal_run/manifest.json` | Source hashes, artifact hash, compiler version |
| `src/botcolosseo/envs/actions.py` | The 13 fixed macro actions and button vectors |
| `src/botcolosseo/envs/types.py` | Actor, privileged, task, and step dataclasses |
| `src/botcolosseo/envs/events.py` | USER-variable snapshots and event decoding |
| `src/botcolosseo/envs/rewards.py` | Capped auditable task reward ledger |
| `src/botcolosseo/envs/single_agent.py` | Single-agent reset/step/close environment |
| `src/botcolosseo/agents/teachers.py` | Five FSM Teachers and RandomLegal baseline |
| `src/botcolosseo/evaluation/m1.py` | Capability evaluator and Wilson intervals |
| `scripts/build_crystal_run.py` | Scenario build CLI |
| `scripts/smoke_crystal_run.py` | Scenario/task/event/video smoke CLI |
| `scripts/evaluate_m1.py` | Frozen 500-episode M1 gate CLI |
| `configs/m1/{train,validation,test}.json` | Frozen disjoint cases |
| `reports/m1/` | Tracked raw results, summary, and provenance |
| `docs/milestones/m1.md` | Public M1 gate/runbook |

---

### Task 1: Add the Minimal Deterministic PWAD Boundary

**Files:**
- Create: `src/botcolosseo/scenarios/__init__.py`
- Create: `src/botcolosseo/scenarios/wad.py`
- Create: `tests/unit/test_wad.py`

**Interfaces:**
- Consumes: ordered `WadLump(name: str, data: bytes)` values.
- Produces: `build_pwad(lumps) -> bytes`, `inspect_pwad(data) -> tuple[WadEntry, ...]`, and `write_pwad(lumps, output_path) -> Path`.

- [ ] **Step 1: Write the failing PWAD tests**

Create `tests/unit/test_wad.py`:

```python
import struct
from pathlib import Path

import pytest

from botcolosseo.scenarios.wad import WadLump, build_pwad, inspect_pwad, write_pwad


def test_build_pwad_has_deterministic_header_and_directory() -> None:
    data = build_pwad((WadLump("MAP01", b""), WadLump("TEXTMAP", b"abc")))

    magic, count, directory_offset = struct.unpack_from("<4sII", data)
    assert (magic, count, directory_offset) == (b"PWAD", 2, 15)
    assert [(entry.name, entry.size) for entry in inspect_pwad(data)] == [
        ("MAP01", 0),
        ("TEXTMAP", 3),
    ]


def test_build_pwad_rejects_invalid_lump_name() -> None:
    with pytest.raises(ValueError, match="1-8 uppercase ASCII"):
        build_pwad((WadLump("too_long_name", b""),))


def test_write_pwad_is_atomic(tmp_path: Path) -> None:
    output = tmp_path / "scenario.wad"

    result = write_pwad((WadLump("MAP01", b""),), output)

    assert result == output.resolve()
    assert output.read_bytes().startswith(b"PWAD")
    assert not tuple(tmp_path.glob(".*.tmp"))
```

- [ ] **Step 2: Run the tests to verify the module is absent**

Run:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_wad.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'botcolosseo.scenarios'`.

- [ ] **Step 3: Implement the PWAD boundary**

Create `src/botcolosseo/scenarios/__init__.py`:

```python
"""Scenario source, build, and manifest utilities."""
```

Create `src/botcolosseo/scenarios/wad.py`:

```python
from __future__ import annotations

import re
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_LUMP_NAME = re.compile(r"[A-Z0-9_]{1,8}")


@dataclass(frozen=True)
class WadLump:
    name: str
    data: bytes

    def __post_init__(self) -> None:
        if _LUMP_NAME.fullmatch(self.name) is None:
            raise ValueError(f"WAD lump name must be 1-8 uppercase ASCII characters: {self.name!r}")


@dataclass(frozen=True)
class WadEntry:
    name: str
    offset: int
    size: int


def build_pwad(lumps: Iterable[WadLump]) -> bytes:
    ordered = tuple(lumps)
    body = bytearray()
    entries: list[WadEntry] = []
    for lump in ordered:
        offset = 12 + len(body)
        body.extend(lump.data)
        entries.append(WadEntry(name=lump.name, offset=offset, size=len(lump.data)))
    directory_offset = 12 + len(body)
    directory = bytearray()
    for entry in entries:
        name = entry.name.encode("ascii").ljust(8, b"\0")
        directory.extend(struct.pack("<II8s", entry.offset, entry.size, name))
    return struct.pack("<4sII", b"PWAD", len(entries), directory_offset) + body + directory


def inspect_pwad(data: bytes) -> tuple[WadEntry, ...]:
    if len(data) < 12:
        raise ValueError("WAD is shorter than its header")
    magic, count, directory_offset = struct.unpack_from("<4sII", data)
    if magic != b"PWAD":
        raise ValueError(f"Expected PWAD magic, got {magic!r}")
    if directory_offset + count * 16 > len(data):
        raise ValueError("WAD directory extends past the file")
    entries = []
    for index in range(count):
        offset, size, raw_name = struct.unpack_from("<II8s", data, directory_offset + index * 16)
        if offset + size > directory_offset:
            raise ValueError("WAD lump extends into or past the directory")
        name = raw_name.rstrip(b"\0").decode("ascii")
        entries.append(WadEntry(name=name, offset=offset, size=size))
    return tuple(entries)


def write_pwad(lumps: Iterable[WadLump], output_path: Path) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary_path.write_bytes(build_pwad(lumps))
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path
```

- [ ] **Step 4: Verify the focused tests and lint**

Run:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_wad.py -v
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src tests
```

Expected: three tests pass and Ruff exits 0.

- [ ] **Step 5: Commit the PWAD boundary**

```bash
git add src/botcolosseo/scenarios tests/unit/test_wad.py
git diff --cached --check
git commit -m "feat: add deterministic PWAD writer"
```

---

### Task 2: Define Regions, Task Variants, and Frozen Split Cases

**Files:**
- Create: `src/botcolosseo/scenarios/regions.py`
- Create: `src/botcolosseo/scenarios/splits.py`
- Create: `assets/scenarios/crystal_run/src/regions.yaml`
- Create: `assets/scenarios/crystal_run/src/task_variants.yaml`
- Create: `tests/unit/test_regions.py`
- Create: `tests/unit/test_splits.py`

**Interfaces:**
- Consumes: YAML region/task definitions and `master_seed`.
- Produces: `RegionGraph`, `TaskKind`, `TaskVariant`, `EpisodeCase`, `generate_split_cases()`, and `write_split_manifests()`.

- [ ] **Step 1: Write failing schema and split tests**

Create `tests/unit/test_regions.py` with tests that load the tracked YAML, assert unique positive IDs, assert every neighbor exists and is reciprocal, assert boundary ownership is deterministic, and assert `shortest_path("home", "center")` differs from the explicit flank route.

Create `tests/unit/test_splits.py` with:

```python
from botcolosseo.scenarios.splits import TaskKind, generate_split_cases


def test_split_cases_are_reproducible_and_disjoint() -> None:
    first = generate_split_cases(master_seed=20260720, cases_per_task=100)
    second = generate_split_cases(master_seed=20260720, cases_per_task=100)

    assert first == second
    assert set(first) == {"train", "validation", "test"}
    assert all(len(first[name]) == 500 for name in first)
    seed_sets = [{case.seed for case in first[name]} for name in first]
    assert seed_sets[0].isdisjoint(seed_sets[1])
    assert seed_sets[0].isdisjoint(seed_sets[2])
    assert seed_sets[1].isdisjoint(seed_sets[2])
    assert {case.task for case in first["test"]} == set(TaskKind)
```

- [ ] **Step 2: Run tests to verify the interfaces are absent**

Run:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_regions.py tests/unit/test_splits.py -v
```

Expected: collection fails because `regions` and `splits` modules do not exist.

- [ ] **Step 3: Create the region and task sources**

Create `regions.yaml` with IDs and inclusive-min/exclusive-max rectangles for
`home`, `upper_route`, `center`, `lower_route`, `flank_west`, `flank_east`,
`shooting_lane`, and `away`; include reciprocal adjacency and ordered waypoint
coordinates for `direct_upper`, `direct_lower`, and `flank` routes.

Create `task_variants.yaml` with exactly these entries:

```yaml
protocol_version: 1
variants:
  navigation: {map: MAP02, timeout_tics: 700}
  pickup: {map: MAP03, timeout_tics: 700}
  return: {map: MAP04, timeout_tics: 900}
  static_hit: {map: MAP05, timeout_tics: 525}
  moving_hit: {map: MAP06, timeout_tics: 700}
```

- [ ] **Step 4: Implement the focused schemas**

In `regions.py`, implement frozen `Bounds`, `Region`, `Route`, and `RegionGraph`
dataclasses. `RegionGraph.from_yaml(path)` must reject overlapping interiors,
unknown/asymmetric neighbors, duplicate IDs/names, empty routes, and waypoints
outside the arena. `region_at(x, y)` uses inclusive lower and exclusive upper
bounds. `shortest_path(start, goal)` uses deterministic breadth-first search
with neighbors sorted by region ID.

In `splits.py`, define:

```python
class TaskKind(StrEnum):
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
```

Generate each split from independent `numpy.random.SeedSequence` children and
allocate non-overlapping integer seed ranges. Serialize cases with sorted keys
and a trailing newline; atomically write `configs/m1/train.json`,
`validation.json`, and `test.json`.

- [ ] **Step 5: Verify and commit region/split contracts**

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_regions.py tests/unit/test_splits.py -v
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src tests
git add src/botcolosseo/scenarios/regions.py src/botcolosseo/scenarios/splits.py assets/scenarios/crystal_run/src/regions.yaml assets/scenarios/crystal_run/src/task_variants.yaml tests/unit/test_regions.py tests/unit/test_splits.py
git diff --cached --check
git commit -m "feat: define Crystal Run tasks and splits"
```

Expected: focused tests and Ruff pass; commit succeeds.

---

### Task 3: Freeze the 13 Macro Actions and Fairness Types

**Files:**
- Create: `src/botcolosseo/envs/actions.py`
- Create: `src/botcolosseo/envs/types.py`
- Create: `tests/unit/test_actions.py`
- Create: `tests/unit/test_observation_types.py`

**Interfaces:**
- Consumes: ViZDoom available-button ordering and raw frame/scalars.
- Produces: `MacroAction`, `ACTION_BUTTONS`, `action_vector()`, `ActorObservation`, `PrivilegedState`, `TaskStep`.

- [ ] **Step 1: Write failing action and leakage tests**

The tests must assert IDs 0–12 match `Plan.md`, every vector has the same length
as `ACTION_BUTTONS`, action 10 presses forward+attack, invalid IDs raise
`ValueError`, frames must be `(84, 84)` `uint8`, and
`dataclasses.fields(ActorObservation)` contains none of `x`, `y`, `angle`,
`region`, `target_x`, `target_y`, `depth`, `labels`, or `automap`.

- [ ] **Step 2: Run tests to verify the modules are absent**

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_actions.py tests/unit/test_observation_types.py -v
```

Expected: collection failure for the absent modules.

- [ ] **Step 3: Implement the exact action enum and vectors**

Define `MacroAction(IntEnum)` with `IDLE=0`, `MOVE_FORWARD=1`,
`MOVE_BACKWARD=2`, `STRAFE_LEFT=3`, `STRAFE_RIGHT=4`, `TURN_LEFT=5`,
`TURN_RIGHT=6`, `FORWARD_TURN_LEFT=7`, `FORWARD_TURN_RIGHT=8`, `ATTACK=9`,
`FORWARD_ATTACK=10`, `TURN_LEFT_ATTACK=11`, and `TURN_RIGHT_ATTACK=12`.

Set `ACTION_BUTTONS` to `(MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT,
TURN_LEFT, TURN_RIGHT, ATTACK)` and implement each vector explicitly as a tuple
of seven floats. `action_vector()` accepts `MacroAction | int`, validates it,
and returns a new list.

- [ ] **Step 4: Implement the frozen data types**

`ActorObservation` fields are `frame`, `health`, `ammo`, `attack_ready`,
`has_core`, `home_score`, `away_score`, `remaining_tics`, and `previous_action`.
Its `__post_init__` enforces the frame contract and makes a read-only contiguous
copy. `PrivilegedState` contains `player_x`, `player_y`, `player_angle`,
`region_name`, `core_x`, `core_y`, `target_x`, `target_y`, `target_alive`, and
`task_phase`. `TaskStep` contains `observation`, `reward`, `terminated`,
`truncated`, and `events`.

- [ ] **Step 5: Verify and commit fairness types**

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_actions.py tests/unit/test_observation_types.py -v
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src tests
git add src/botcolosseo/envs/actions.py src/botcolosseo/envs/types.py tests/unit/test_actions.py tests/unit/test_observation_types.py
git diff --cached --check
git commit -m "feat: add fair M1 observation and action contracts"
```

Expected: tests and Ruff pass; commit succeeds.

---

### Task 4: Decode Protocol-v1 Events and Cap Task Rewards

**Files:**
- Create: `src/botcolosseo/envs/events.py`
- Create: `src/botcolosseo/envs/rewards.py`
- Create: `tests/unit/test_events.py`
- Create: `tests/unit/test_rewards.py`

**Interfaces:**
- Consumes: ordered raw values for `USER1`–`USER20`, episode/decision IDs, and region membership.
- Produces: `ProtocolSnapshot`, `EpisodeEvent`, `EventDecoder.decode()`, and `RewardLedger.apply()`.

- [ ] **Step 1: Write failing event decoder tests**

Create a `snapshot(**overrides)` helper whose protocol version is 1, counters
are zero, coordinates are fixed-point zero, and reserved `USER16` is zero.
Tests must prove that a pickup counter delta emits exactly one `PICKUP`, reading
the same snapshot again emits nothing, simultaneous pickup and score deltas emit
two ordered events, a decreasing counter raises `EventProtocolError`, a delta
above one raises it, a protocol mismatch raises it, nonzero USER16 raises it,
and `reset()` permits counters to return to zero for the next episode.

- [ ] **Step 2: Write failing reward positive, negative, cap, and reset tests**

Use immutable events to prove:

- pickup pays `0.25` once and is capped at one payment per episode;
- score pays `1.0`, valid hit pays `0.05` up to five times;
- region transition pays `0.01` only if graph distance to the task target falls;
- repeated snapshots and sideways region movement pay zero;
- `reset()` clears all caps and progress history.

- [ ] **Step 3: Run tests to verify both interfaces are absent**

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_events.py tests/unit/test_rewards.py -v
```

Expected: collection failures for `events` and `rewards`.

- [ ] **Step 4: Implement `events.py`**

Define:

```python
PROTOCOL_VERSION = 1
COUNTER_FIELDS = (
    "pickup_count",
    "drop_count",
    "score_count",
    "valid_hit_count",
    "death_count",
    "respawn_count",
    "core_return_count",
    "task_success_count",
)


class EventType(StrEnum):
    PICKUP = "pickup"
    DROP = "drop"
    SCORE = "score"
    VALID_HIT = "valid_hit"
    DEATH = "death"
    RESPAWN = "respawn"
    CORE_RETURN = "core_return"
    TASK_SUCCESS = "task_success"
    REGION_TRANSITION = "region_transition"


@dataclass(frozen=True)
class EpisodeEvent:
    episode_id: int
    engine_tic: int
    decision_index: int
    type: EventType
    subject: str = "agent"
    region_from: str | None = None
    region_to: str | None = None
    value: float = 1.0
```

`ProtocolSnapshot.from_values(values)` requires exactly 20 values, converts
USER17–USER20 with `vzd.doom_fixed_to_float`, and maps USER5–USER12 to the
counter fields. `EventDecoder` stores the previous snapshot and region,
validates protocol/reserved/counter invariants, emits counter events in
`COUNTER_FIELDS` order followed by one region transition, and updates its state
only after the complete snapshot validates.

- [ ] **Step 5: Implement `rewards.py`**

Define frozen `RewardConfig` defaults `pickup=0.25`, `score=1.0`,
`valid_hit=0.05`, `progress=0.01`, with caps `pickup=1`, `score=3`,
`valid_hit=5`, `progress=25`. `RewardLedger` owns per-event counts and the
previous shortest graph distance. `apply(events, target_region)` sums only
eligible values, enforces caps, and never reads raw ACS counters or positions.

- [ ] **Step 6: Verify and commit the audit boundary**

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_events.py tests/unit/test_rewards.py -v
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src tests
git add src/botcolosseo/envs/events.py src/botcolosseo/envs/rewards.py tests/unit/test_events.py tests/unit/test_rewards.py
git diff --cached --check
git commit -m "feat: add auditable M1 event and reward protocol"
```

Expected: focused tests and Ruff pass; commit succeeds.

---

### Task 5: Add Crystal Run UDMF, ACS, CFG, and Reproducible Build

**Files:**
- Create: `assets/scenarios/crystal_run/src/map.udmf`
- Create: `assets/scenarios/crystal_run/src/crystal_run.acs`
- Create: `assets/scenarios/crystal_run/crystal_run.cfg`
- Create: `assets/scenarios/crystal_run/README.md`
- Create: `assets/scenarios/crystal_run/LICENSES.md`
- Create: `src/botcolosseo/scenarios/build.py`
- Create: `src/botcolosseo/cli/build_scenario.py`
- Create: `scripts/build_crystal_run.py`
- Create: `tests/unit/test_scenario_build.py`
- Create: `tests/integration/test_scenario_build.py`
- Produce: `assets/scenarios/crystal_run/crystal_run.wad`
- Produce: `assets/scenarios/crystal_run/manifest.json`

**Interfaces:**
- Consumes: UDMF/ACS/YAML sources, ACC executable/include directory, and `WadLump`.
- Produces: `BuildSettings`, `build_crystal_run(settings) -> ScenarioManifest`, and a six-map tracked PWAD.

- [ ] **Step 1: Write failing build-unit tests**

Tests must use a fake subprocess runner and temporary source tree to prove ACC
resolution priority (`--acc`, `ACC_PATH`, `PATH`), wrapper generation with
`#define BOTC_TASK_ID <1..6>`, lump order repeated as
`MAPxx,TEXTMAP,BEHAVIOR,SCRIPTS,ENDMAP`, sorted SHA-256 manifest keys, atomic
replacement, and cleanup after compiler failure.

- [ ] **Step 2: Run the unit test to establish RED**

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_scenario_build.py -v
```

Expected: collection fails because `botcolosseo.scenarios.build` is absent.

- [ ] **Step 3: Create the UDMF geometry source**

Create a UDMF `namespace = "zdoom"` map with a `1536×1024` outer sector,
ceiling height 128, light level 192, Freedoom-compatible `FLOOR0_1`, `FLAT4`,
and `BRICK9` textures. Place player start type 1 at `(-640, 0)` facing east.
Place blocking column things in two staggered bands so the upper and lower
direct lanes remain readable and the southern flank requires the longest
waypoint sequence. The four boundary linedefs are one-sided and blocking. All
things enable skills 1–5 and single-player.

The exact source is validated structurally rather than by textual snapshot:
one player start, four outer vertices, four boundary linedefs, four sidedefs,
one sector, at least twelve blocking route-marker columns, and no commercial
texture name outside the scenario allowlist.

- [ ] **Step 4: Create the ACS protocol source**

Create `crystal_run.acs` beginning with:

```c
#include "zcommon.acs"

#ifndef BOTC_TASK_ID
#define BOTC_TASK_ID 1
#endif

global int 0:reward;
global int 1:protocol_version;
global int 2:engine_tic;
global int 3:task_phase;
global int 4:core_state;
global int 5:pickup_count;
global int 6:drop_count;
global int 7:score_count;
global int 8:valid_hit_count;
global int 9:death_count;
global int 10:respawn_count;
global int 11:core_return_count;
global int 12:task_success_count;
global int 13:target_state;
global int 14:home_score;
global int 15:away_score;
global int 16:reserved_zero;
global int 17:core_x;
global int 18:core_y;
global int 19:target_x;
global int 20:target_y;
```

Use constants `CORE_TID=100`, `TARGET_TID=200`, and protocol version 1. Script
`OPEN` resets all globals, initializes task-specific core/target state, and
starts a one-tic monitor. Script `ENTER` removes the fist, gives pistol/ammo,
and for return gives one `RedCard`. The monitor:

- increments engine tic;
- publishes core and target coordinates with `GetActorX/Y`;
- detects first `RedCard` possession as pickup;
- detects carried core entering home rectangle as score and task success;
- compares target health to its prior value for valid-hit events;
- detects player health reaching zero once;
- returns an uncollected core after 350 tics;
- calls `Exit_Normal(0)` on task success for MAP02–MAP06;
- delays one tic and repeats.

Task setup uses deterministic ACS `Random()` after ViZDoom seed assignment.
Navigation teleports the player among allowed starts and sets success on entry
to center. Pickup spawns `RedCard` at one of three center candidates. Return
teleports a core-carrying player to one of three away starts. Static hit spawns
a one-hit-point `Zombieman` with speed zero. Moving hit spawns a `Zombieman`
with normal speed and `Thing_Hate(TARGET_TID, 0, 6)`.

- [ ] **Step 5: Create the ViZDoom CFG**

Create `crystal_run.cfg` with relative `doom_scenario_path = crystal_run.wad`,
default `doom_map = map01`, `episode_start_time = 10`,
`episode_timeout = 2100`, `screen_resolution = RES_320X240`, grayscale format,
HUD/crosshair/decals/particles disabled, weapon enabled, and the seven buttons
from `ACTION_BUTTONS`. Expose `HEALTH`, `SELECTED_WEAPON_AMMO`, `ATTACK_READY`,
`POSITION_X`, `POSITION_Y`, `ANGLE`, `HITCOUNT`, and `USER1`–`USER20`.

- [ ] **Step 6: Implement the builder**

`BuildSettings` contains `source_dir`, `output_wad`, `manifest_path`,
`acc_path`, and `acc_include`. For each MAP01–MAP06, create a wrapper ACS file,
invoke:

```text
acc -i <include> <wrapper.acs> <behavior.o>
```

Require exit code 0 and nonempty behavior output. Include the readable ACS
source as `SCRIPTS`. Build the WAD with Task 1, inspect its directory, and write
a manifest containing schema version 1, protocol version 1, UTC build date,
ACC first version line, source SHA-256 values, WAD SHA-256, lump names, and map
markers. Exclude the build date from reproducibility comparison: two builds
must produce identical WAD bytes and identical hash-bearing manifest fields.

- [ ] **Step 7: Add CLI and source licensing**

`python scripts/build_crystal_run.py --check` builds into a temporary directory
and exits nonzero if bytes differ from the tracked WAD. Default mode atomically
updates the WAD and manifest. `--acc` and `--acc-include` override discovery.
Document ACC commands and source/artifact roles in the scenario README.
`LICENSES.md` states that map/script sources are MIT, Freedoom assets remain BSD
3-Clause, only names/references are used, and no commercial assets are bundled.

- [ ] **Step 8: Build the tracked artifact with real ACC**

Run:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python scripts/build_crystal_run.py --acc /home/wencong/.local/bin/acc --acc-include /home/wencong/.local/src/acc-1.60
```

Expected: six ACC compilations succeed; WAD and manifest are created.

- [ ] **Step 9: Write and run real build integration tests**

The integration test runs `--check`, then loads every MAP01–MAP06 with a raw
`DoomGame`: load the CFG, override `doom_map`, apply the same M0 silent/headless
settings before `init()`, assert a nonempty grayscale frame and protocol version
1, and close in `finally`. Task 6 later folds the map override into
`GameSettings`; Task 5 must not preempt that tested interface change.

```bash
timeout 90s /home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/integration/test_scenario_build.py -v
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src tests scripts
```

Expected: deterministic rebuild and all six map loads pass; Ruff exits 0.

- [ ] **Step 10: Commit the reproducible scenario**

```bash
git add assets/scenarios/crystal_run src/botcolosseo/scenarios/build.py src/botcolosseo/cli/build_scenario.py scripts/build_crystal_run.py tests/unit/test_scenario_build.py tests/integration/test_scenario_build.py
git diff --cached --check
git commit -m "feat: add reproducible Crystal Run scenario"
```

Expected: the small WAD is tracked, temporary build files are absent, and commit succeeds.

---

### Task 6: Implement the Single-Agent Task Environment

**Files:**
- Modify: `src/botcolosseo/envs/vizdoom_game.py`
- Create: `src/botcolosseo/envs/single_agent.py`
- Create: `tests/unit/test_single_agent.py`
- Modify: `tests/unit/test_vizdoom_game.py`

**Interfaces:**
- Consumes: `GameSettings`, `TaskKind`, action vectors, protocol snapshots, region graph, and reward ledger.
- Produces: `SingleAgentTaskEnv.reset(seed, task)` and `SingleAgentTaskEnv.step(action) -> TaskStep`.

- [ ] **Step 1: Extend the failing factory test**

Add a test proving `GameSettings(..., doom_map="MAP03")` calls
`game.set_doom_map("MAP03")` after config load and before `game.init()`. Add
`doom_map: str | None = None` to the expected public settings contract.

- [ ] **Step 2: Write failing environment lifecycle tests**

Use a `FakeGame` with deterministic frames and variable vectors to prove:

- reset seeds before creating the game, starts an episode, resets event/reward
  state, and returns `(ActorObservation, ResetInfo)`;
- the observation frame is resized to `(84, 84)` grayscale and contains only
  legal scalars;
- step maps macro action 10 to forward+attack, advances exactly `frame_skip`,
  decodes events once, and distinguishes `terminated` from `truncated`;
- task success terminates, decision limit truncates, and engine completion
  without task success truncates;
- a missing state, invalid protocol, or engine exception closes the game;
- `close()` is idempotent and reset closes any earlier game first.

- [ ] **Step 3: Run tests to establish RED**

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_vizdoom_game.py tests/unit/test_single_agent.py -v
```

Expected: missing `doom_map` and `single_agent` failures.

- [ ] **Step 4: Extend `create_game` minimally**

After a successful `load_config`, validate `doom_map` against
`MAP01`–`MAP99` with a full-match regex and call `set_doom_map` when present.
Keep all M0 silent/headless settings and failure cleanup unchanged.

- [ ] **Step 5: Implement `SingleAgentTaskEnv`**

Constructor parameters are:

```python
def __init__(
    self,
    *,
    config_path: Path,
    region_graph: RegionGraph,
    frame_skip: int = 4,
    max_decisions: int = 225,
    game_builder: Callable[[GameSettings], Any] = create_game,
) -> None:
```

`reset(seed, task)` selects the MAP marker from the frozen variant map, creates
the game, starts an episode, reads the first state and 20 user variables,
initializes decoder/ledger, and returns an observation plus frozen `ResetInfo`
containing episode ID, seed, task, and scenario hash. `step()` makes one action,
reads the new state if available, emits events, applies reward, and returns a
`TaskStep`. When ViZDoom finishes before a final frame, reuse the last legal
observation only for the returned terminal step; never fabricate privileged
state or events.

Resize frames with OpenCV `INTER_AREA`; reject non-grayscale buffers instead of
silently accepting labels/RGB. Read exact player position and target coordinates
only into `PrivilegedState`. Expose privileged state through
`teacher_state() -> PrivilegedState`, not through reset/step observation data.

- [ ] **Step 6: Verify environment tests and regression suite**

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_vizdoom_game.py tests/unit/test_single_agent.py -v
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit -v
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src tests
```

Expected: focused and full unit tests pass; Ruff exits 0.

- [ ] **Step 7: Commit the environment boundary**

```bash
git add src/botcolosseo/envs/vizdoom_game.py src/botcolosseo/envs/single_agent.py tests/unit/test_vizdoom_game.py tests/unit/test_single_agent.py
git diff --cached --check
git commit -m "feat: add single-agent Crystal Run environment"
```

---

### Task 7: Implement Five Deterministic Teachers and RandomLegal

**Files:**
- Create: `src/botcolosseo/agents/__init__.py`
- Create: `src/botcolosseo/agents/teachers.py`
- Create: `tests/unit/test_teachers.py`

**Interfaces:**
- Consumes: `PrivilegedState`, region routes, task kind, previous action, and deterministic RNG.
- Produces: `Teacher.act(state) -> MacroAction`, `Teacher.reset(seed, task)`, five named Teachers, and `RandomLegal`.

- [ ] **Step 1: Write failing FSM tests**

Use synthetic privileged states to assert:

- `FixedRouteTeacher` turns toward the next waypoint when angular error exceeds
  12 degrees, otherwise moves forward, and advances waypoints within 48 units;
- `ObjectiveFirstTeacher` transitions `SEARCH_CORE -> PICKUP -> RETURN_BASE`
  based on core possession/task phase and chooses the shortest region path;
- `AggressiveScriptTeacher` attacks only when target is alive, within 512 units,
  and angular error is at most 8 degrees; otherwise it turns toward the target;
- `DefensiveScriptTeacher` returns to its hold waypoint, idles inside 32 units
  with no live target, and delegates valid engagement to the aggressive rule;
- `EvasiveReturnTeacher` chooses the named flank waypoints while carrying core
  and alternates deterministic evasive turns every eight decisions;
- same seed/state sequence gives identical actions for every Teacher;
- `RandomLegal` samples only IDs 0–12 and reproduces a seed sequence.

- [ ] **Step 2: Run the tests to establish RED**

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_teachers.py -v
```

Expected: collection fails because `botcolosseo.agents` is absent.

- [ ] **Step 3: Implement shared geometry helpers and the Teacher protocol**

Define `Teacher(Protocol)` with `name`, `reset(seed, task)`, and `act(state)`.
Pure helpers compute Euclidean distance, normalized signed angular error in
degrees, and turn/advance actions. Never read Actor frames in M1 Teachers.

Use a private `_WaypointFollower` with a waypoint index and 48-unit arrival
tolerance. It returns combined forward-turn actions for errors from 12–45
degrees, pure turn above 45 degrees, and forward inside 12 degrees.

- [ ] **Step 4: Implement all named policies**

Implement exactly `FixedRouteTeacher`, `ObjectiveFirstTeacher`,
`AggressiveScriptTeacher`, `DefensiveScriptTeacher`, `EvasiveReturnTeacher`,
and `RandomLegal`. Each owns its RNG created by `np.random.default_rng(seed)`;
module-global randomness is forbidden. Expose `TEACHER_REGISTRY` keyed by
`fixed_route`, `objective_first`, `aggressive_script`, `defensive_script`, and
`evasive_return`; omit RandomLegal from that registry.

- [ ] **Step 5: Verify and commit Teachers**

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_teachers.py -v
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src tests
git add src/botcolosseo/agents tests/unit/test_teachers.py
git diff --cached --check
git commit -m "feat: add deterministic Crystal Run Teachers"
```

Expected: Teacher tests and Ruff pass; commit succeeds.

---

### Task 8: Add Real Scenario, Event, Negative, Replay, and Video Gates

**Files:**
- Create: `src/botcolosseo/cli/smoke_crystal_run.py`
- Create: `scripts/smoke_crystal_run.py`
- Create: `tests/integration/test_crystal_run_events.py`
- Create: `tests/integration/test_single_agent_tasks.py`
- Create: `tests/integration/test_crystal_run_replay.py`

**Interfaces:**
- Consumes: tracked WAD/CFG, `SingleAgentTaskEnv`, Teachers, and M0 MP4 writer.
- Produces: structured JSON smoke evidence and real regression tests for every M1 task/event path.

- [ ] **Step 1: Write the failing real tests**

Tests use bounded scripted action traces and assert:

- MAP02 emits one region transition and one task success after reaching center;
- MAP03 emits pickup only after inventory changes;
- MAP04 emits score only while carrying core inside home;
- MAP05/06 emit valid hit only when target health falls;
- standing still outside core/home and firing away from the target emits none of
  pickup, score, or valid hit;
- all five task variants distinguish success termination from time truncation;
- same seed, task, and Teacher produce identical terminal outcome and ordered
  event-type sequence twice;
- required MP4 is nonempty and game process cleanup succeeds.

Mark all tests `@pytest.mark.integration` and give each test
`@pytest.mark.timeout(30)`.

- [ ] **Step 2: Run a focused test to establish RED**

```bash
timeout 35s /home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/integration/test_single_agent_tasks.py::test_navigation_teacher_completes -v
```

Expected: failure because the Crystal Run smoke CLI/real harness is absent or
the uncalibrated Teacher does not yet complete.

- [ ] **Step 3: Implement the smoke harness and CLI**

`run_crystal_smoke(task, teacher_name, seed, record_path)` resets the task,
runs until terminated/truncated, records first-episode frames optionally, and
returns a frozen summary with success, decisions, total reward, event counts,
event types, first-frame shape, scenario hash, and video path/error. Always
close in `finally`.

The CLI accepts `--task`, `--teacher`, `--seed`, `--record`, and
`--require-video`; prints sorted JSON; and exits zero only on task success and,
when requested, successful video.

- [ ] **Step 4: Calibrate only declarative waypoints/tolerances**

Run each focused integration trace. Adjust `regions.yaml` waypoints, ACS spawn
coordinates, and documented Teacher tolerances only. Do not add privileged
fields, new actions, hidden teleport-on-step behavior, event shortcuts, or a
different map. Every calibration change must retain unit tests and negative
event tests.

- [ ] **Step 5: Run complete integration and video gates**

```bash
timeout 180s /home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/integration -v
timeout 30s /home/wencong/miniconda3/envs/botcolosseo/bin/python scripts/smoke_crystal_run.py --task navigation --teacher fixed_route
timeout 30s /home/wencong/miniconda3/envs/botcolosseo/bin/python scripts/smoke_crystal_run.py --task pickup --teacher objective_first --record videos/m1-pickup.mp4 --require-video
test -s videos/m1-pickup.mp4
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src tests scripts
```

Expected: all integration tests pass, both smokes succeed, video is nonempty,
and Ruff passes.

- [ ] **Step 6: Commit real M1 gates**

```bash
git add src/botcolosseo/cli/smoke_crystal_run.py scripts/smoke_crystal_run.py tests/integration assets/scenarios/crystal_run/src
git diff --cached --check
git commit -m "test: add real Crystal Run task and event gates"
```

Expected: only source calibration changes are staged; transient video remains ignored.

---

### Task 9: Implement and Run the Quantitative 500-Episode M1 Gate

**Files:**
- Create: `src/botcolosseo/evaluation/__init__.py`
- Create: `src/botcolosseo/evaluation/m1.py`
- Create: `src/botcolosseo/cli/evaluate_m1.py`
- Create: `scripts/evaluate_m1.py`
- Create: `tests/unit/test_m1_evaluation.py`
- Produce: `configs/m1/train.json`
- Produce: `configs/m1/validation.json`
- Produce: `configs/m1/test.json`
- Produce: `reports/m1/episodes.csv`
- Produce: `reports/m1/summary.json`
- Produce: `reports/m1/manifest.json`

**Interfaces:**
- Consumes: frozen split cases, scenario manifest, task/Teacher mapping, and `SingleAgentTaskEnv`.
- Produces: `CapabilityResult`, Wilson intervals, per-episode CSV, summary JSON, provenance manifest, and process exit gate.

- [ ] **Step 1: Write failing evaluator tests**

Test `wilson_interval(successes, trials, confidence=0.95)` against known values
for 0/100, 75/100, and 100/100. Use a fake episode runner to prove the evaluator
runs every case exactly once, maps navigation→fixed route, pickup→objective
first, return→evasive return, and both hit tasks→aggressive; computes success
rates; counts protocol inconsistencies; writes deterministic row ordering; and
fails if a threshold is missed or inconsistencies are nonzero.

- [ ] **Step 2: Run tests to establish RED**

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_m1_evaluation.py -v
```

Expected: collection fails because `botcolosseo.evaluation.m1` is absent.

- [ ] **Step 3: Implement evaluator types and gate rules**

Define:

```python
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
```

`CapabilityResult` stores task, Teacher, successes, trials, success rate,
Wilson lower/upper 95% bounds, threshold, and passed. The pass comparison uses
the empirical success rate, while the interval is reported for uncertainty.
`EvaluationSummary.passed` requires all five capabilities and zero protocol
inconsistencies. Do not tune against test rows inside the evaluator.

- [ ] **Step 4: Implement the CLI and frozen manifests**

`scripts/evaluate_m1.py --split test --output reports/m1` loads exactly the
tracked test manifest, refuses duplicate/overlapping seeds, runs all 500 cases
serially, prints progress every 25 episodes, atomically writes evidence, prints
the final JSON, and returns nonzero on failure. `--max-cases` is allowed only
with `--development`; development summaries are visibly marked and cannot pass
the official gate.

Generate all three split JSON files once with master seed `20260720`, 100 cases
per task per split, then commit them before the official test run.

- [ ] **Step 5: Verify unit tests and run a 10-case development check**

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_m1_evaluation.py -v
timeout 60s /home/wencong/miniconda3/envs/botcolosseo/bin/python scripts/evaluate_m1.py --split validation --max-cases 10 --development --output /tmp/botcolosseo-m1-dev
```

Expected: unit tests pass; ten development cases finish; output is marked
`official: false` and cannot be confused with the gate.

- [ ] **Step 6: Commit evaluator and frozen manifests before test evaluation**

```bash
git add src/botcolosseo/evaluation src/botcolosseo/cli/evaluate_m1.py scripts/evaluate_m1.py tests/unit/test_m1_evaluation.py configs/m1
git diff --cached --check
git commit -m "feat: add frozen M1 capability evaluator"
```

- [ ] **Step 7: Run the official 500-episode gate synchronously**

Run in the foreground; do not use `nohup`:

```bash
timeout 1800s /home/wencong/miniconda3/envs/botcolosseo/bin/python scripts/evaluate_m1.py --split test --output reports/m1
```

Expected: exactly 500 rows, 100 per task; navigation/pickup/return are at least
95%, static hit at least 90%, moving hit at least 75%, protocol inconsistencies
are zero, and exit code is 0. If runtime approaches the timeout, optimize only
process reuse and render settings without changing cases or semantics. If this
must become a manual background job, append the exact command and artifact
checks to `script.md` and stop under the user's explicit `nohup` rule.

- [ ] **Step 8: Verify evidence integrity and commit results**

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -c "import csv,json,pathlib; p=pathlib.Path('reports/m1'); rows=list(csv.DictReader((p/'episodes.csv').open())); summary=json.loads((p/'summary.json').read_text()); assert len(rows)==500; assert summary['passed'] is True; assert summary['protocol_inconsistencies']==0"
git add reports/m1
git diff --cached --check
git commit -m "results: record Milestone 1 capability gate"
```

Expected: evidence checks and commit succeed.

---

### Task 10: Package the M1 Visual Story and Public Documentation

**Files:**
- Create: `src/botcolosseo/demo/m1_showcase.py`
- Create: `src/botcolosseo/demo/__init__.py`
- Create: `scripts/render_m1_showcase.py`
- Create: `tests/unit/test_m1_showcase.py`
- Create: `docs/assets/m1-arena.png`
- Create: `docs/assets/m1-teacher-montage.mp4`
- Create: `docs/milestones/m1.md`
- Modify: `README.md`
- Modify: `Plan.md`

**Interfaces:**
- Consumes: region graph, real task frames/events, and official M1 summary.
- Produces: a labeled arena figure, short Teacher montage, truthful README status, and exact M1 runbook.

- [ ] **Step 1: Write failing showcase-data tests**

Test that `load_m1_summary()` refuses non-official or failed evidence, that the
arena plot contains all named routes/regions, that montage frame selection is
deterministic and capped, and that overlay text contains task, Teacher, event,
and success without privileged coordinates.

- [ ] **Step 2: Run tests to establish RED**

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_m1_showcase.py -v
```

Expected: collection fails because `botcolosseo.demo.m1_showcase` is absent.

- [ ] **Step 3: Implement deterministic arena and montage rendering**

Use Matplotlib with a fixed 12×7 inch figure, equal axes, stable colors, region
labels, route polylines, home/core/target markers, and a legend. Save PNG at
150 DPI with fixed metadata. Build a montage from one successful held-out case
for navigation, pickup, return, static hit, and moving hit; sample at most 40
frames per task, add a top overlay with OpenCV, and encode at 10 FPS with the M0
atomic video writer. No position or target-coordinate values appear in overlays.

- [ ] **Step 4: Generate and validate showcase assets**

```bash
timeout 180s /home/wencong/miniconda3/envs/botcolosseo/bin/python scripts/render_m1_showcase.py
test -s docs/assets/m1-arena.png
test -s docs/assets/m1-teacher-montage.mp4
ffprobe -v error -show_entries format=duration,size -of json docs/assets/m1-teacher-montage.mp4
```

Expected: both files are nonempty; montage duration is at most 25 seconds and
size is at most 10 MB.

- [ ] **Step 5: Write the M1 runbook and truthful README**

`docs/milestones/m1.md` includes build prerequisites, deterministic rebuild,
all unit/integration commands, task smoke examples, the official evaluator,
thresholds, artifact descriptions, and failure diagnosis. README adds the arena
image, official five-row table sourced exactly from `summary.json`, links the
montage and raw evidence, and clearly states that M2 learning/multiplayer and
the three learned style Bots are not implemented yet.

Update stale `Plan.md` environment facts: repository and M0 are complete,
ViZDoom real-host gates pass, and M1 evidence is recorded. Do not alter the
approved route, fairness boundary, or later milestone thresholds.

- [ ] **Step 6: Verify and commit public packaging**

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_m1_showcase.py -v
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src tests scripts
git diff --check
git add src/botcolosseo/demo scripts/render_m1_showcase.py tests/unit/test_m1_showcase.py docs/assets docs/milestones/m1.md README.md Plan.md
git diff --cached --check
git commit -m "docs: publish Milestone 1 evidence"
```

Expected: tests/lint/diff checks pass and packaging commit succeeds.

---

### Task 11: Run the Complete M1 Acceptance Audit

**Files:**
- Modify only if evidence exposes a defect; any fix uses a new failing regression test.

**Interfaces:**
- Consumes: all tracked M1 sources, WAD, frozen manifests, results, and public artifacts.
- Produces: evidence that every M1 requirement is met before automatically entering M2 design.

- [ ] **Step 1: Verify environment and deterministic scenario build**

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python scripts/check_env.py
/home/wencong/miniconda3/envs/botcolosseo/bin/python scripts/build_crystal_run.py --check --acc /home/wencong/.local/bin/acc --acc-include /home/wencong/.local/src/acc-1.60
```

Expected: environment exits 0 and tracked WAD exactly matches rebuilt bytes.

- [ ] **Step 2: Run all static and automated tests**

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pip check
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src tests scripts
timeout 300s /home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest -v
```

Expected: no broken requirements, Ruff errors, failures, hangs, or skipped
required ACC/ViZDoom tests.

- [ ] **Step 3: Re-run public smoke and artifact checks**

```bash
timeout 30s /home/wencong/miniconda3/envs/botcolosseo/bin/python scripts/smoke_crystal_run.py --task navigation --teacher fixed_route
timeout 30s /home/wencong/miniconda3/envs/botcolosseo/bin/python scripts/smoke_crystal_run.py --task moving_hit --teacher aggressive_script --record videos/m1-final.mp4 --require-video
test -s videos/m1-final.mp4
test -s docs/assets/m1-arena.png
test -s docs/assets/m1-teacher-montage.mp4
```

Expected: both task smokes succeed and all visual artifacts are nonempty.

- [ ] **Step 4: Audit official quantitative evidence against the frozen test manifest**

Run an evidence verifier that recomputes row counts, unique keys, split
disjointness, per-task successes, rates, Wilson intervals, thresholds, protocol
inconsistencies, WAD hash, Git commit, and summary fields directly from
`episodes.csv`, frozen manifests, and tracked artifact. Expected: all checks
pass without trusting summary booleans alone.

- [ ] **Step 5: Confirm repository boundary**

```bash
git status --short
git diff --check
git check-ignore -v _vizdoom.ini '盛文聪_中国科学院自动化研究所_强化学习.pdf' videos/m1-final.mp4 build/
git log --oneline --decorate -16
```

Expected: clean worktree; personal/runtime/generated files ignored; commits map
cleanly to M1 tasks.

- [ ] **Step 6: Record the gate transition**

Append a dated `M1 passed` entry to `docs/milestones/m1.md` containing exact
test counts, capability rates, artifact hashes, and final commit. Commit only
that evidence update as:

```bash
git add docs/milestones/m1.md
git diff --cached --check
git commit -m "chore: close Milestone 1 gate"
```

After this commit, automatically begin the separately scoped M2 brainstorming
and design cycle. Stop only if evidence fails, a manual `nohup` run becomes
necessary, or M2 requires changing the approved technical route.

## Plan Self-Review Record

- Spec coverage: source-first build, one geometry source, six task markers,
  USER1–USER20 protocol, fairness types, 13 actions, five Teachers, negative
  event checks, deterministic splits, 500-episode quantitative gate, visual
  packaging, licensing, and automatic milestone transition each map to tasks.
- Scope: no synchronous duel, demonstration dataset, BC, PPO, styles, league,
  or difficulty implementation appears in M1.
- Type consistency: `TaskKind`, `ActorObservation`, `PrivilegedState`,
  `ProtocolSnapshot`, `EpisodeEvent`, `SingleAgentTaskEnv`, and Teacher names are
  introduced once and consumed consistently.
- Reproducibility: the official test manifest is committed before evaluation;
  evidence records source/WAD/Git hashes and is verified from raw rows.
- Placeholder scan: implementation steps name exact files, commands, contracts,
  failure behavior, and expected results; calibration is restricted to
  declarative geometry/waypoints and cannot change semantics.
