# Bot Colosseo M1 Crystal Run Design

**Status:** Approved route, implementation-ready after self-review

**Milestone:** M1 — single-instance task prototype

**Authority:** `Plan.md`, the verified M0 interfaces, and the accepted M1 gates

## 1. Outcome

M1 delivers one source-controlled Crystal Run Arena that can be built without a
GUI, loaded by ViZDoom, and exercised through a typed single-agent task API. It
also delivers an auditable event protocol, deterministic train/validation/test
configuration splits, and five rule-based Teachers that demonstrate the
capabilities required before multiplayer or learning begins.

M1 does not implement synchronous 1v1, behavior cloning, PPO, style shaping,
league training, or difficulty control. Those remain behind later gates.

## 2. Accepted Gate

Every gated result uses 100 held-out test seeds that are disjoint from
development and validation seeds:

| Capability | Required success rate |
|---|---:|
| Navigation to a requested region | at least 95% |
| Core pickup | at least 95% |
| Core return to base | at least 95% |
| Valid hit on a static target | at least 90% |
| Valid hit on a moving target | at least 75% |

Across all gate episodes, the decoded `EpisodeEvent` stream must have zero
protocol inconsistencies. A repeated run with the same seed, task, and Teacher
must reproduce the same terminal outcome and event-type sequence.

The single-agent full-objective completion rate is reported as an informative
metric in M1, not used as a learning claim or as a substitute for the five
capability gates.

## 3. Scenario Build Decision

Three approaches were considered:

1. **Source-first deterministic build (selected).** Keep UDMF, ACS, region, and
   configuration sources as reviewable text; compile ACS with ACC 1.60; package
   the WAD with a small tested Python writer; commit the small runnable WAD and a
   manifest containing source and artifact hashes. SLADE remains an optional
   visual inspection tool.
2. **GUI-authored binary WAD.** Fast for interactive editing, but binary diffs
   are opaque, headless reproduction is weak, and SLADE cannot initialize GTK
   in the target SSH session.
3. **Third-party Python WAD dependency.** A library such as Omgifol reduces a
   small amount of binary packing code, but adds an unnecessary dependency and
   still does not compile ACS.

The selected approach gives reviewers readable map logic and gives users a WAD
that runs immediately. The build is deterministic and does not require
commercial Doom assets. This matches ViZDoom's documented custom-scenario
boundary: UDMF map and ACS logic in a WAD, with runtime settings in a CFG.

## 4. Source and Artifact Layout

```text
assets/scenarios/crystal_run/
├── README.md
├── LICENSES.md
├── crystal_run.cfg
├── crystal_run.wad                 # reproducible runnable artifact
├── manifest.json                   # hashes, ACC version, protocol version
└── src/
    ├── map.udmf                    # single arena geometry source
    ├── crystal_run.acs             # task state and cumulative event counters
    ├── regions.yaml                # region rectangles, IDs, and adjacency
    └── task_variants.yaml          # map marker, setup, timeout, target behavior
```

`scripts/build_crystal_run.py` resolves ACC from `--acc`, `ACC_PATH`, then
`PATH`; validates version and inputs; compiles ACS; writes a temporary WAD;
loads it in ViZDoom; and atomically replaces the tracked artifact and manifest.
Temporary build output stays under ignored `build/`.

The WAD packer supports only the lumps M1 needs. It is not a general Doom asset
library. Unit tests cover header, offsets, names, ordering, deterministic bytes,
and failure cleanup.

## 5. Arena and Task Variants

The arena is compact and symmetric. Two exposed routes and one longer flank
connect each base to the center. Stable rectangular regions identify both
bases, center, route segments, flank, and shooting lane. Geometry is authored
once in `map.udmf`.

The build emits task map markers from the same geometry and compiled behavior:

| Marker | Task | Deterministic setup |
|---|---|---|
| `MAP01` | full objective | player at home, core at seeded center candidate |
| `MAP02` | navigation | player and requested target region are seed-selected |
| `MAP03` | pickup | player starts away from a seeded core position |
| `MAP04` | return | player starts carrying the core away from home |
| `MAP05` | static hit | stationary target in the shooting lane |
| `MAP06` | moving hit | target patrols a deterministic seeded path |

These are task configurations of one arena, not six independently maintained
maps. The build duplicates the UDMF map lump under explicit markers so task
initialization does not depend on undocumented console-variable behavior.

The prototype uses a Freedoom-compatible built-in key inventory item as the
energy core. Its semantics are the stable contract; custom artwork can replace
the visual later without changing events or task APIs.

## 6. Event Protocol

ACS exposes episode-local monotonic counters and current state through ViZDoom
`USER1`–`USER20` variables. `USER1` is always the protocol version. Global
variable 0 remains reserved for ViZDoom reward and is not used as an event
channel.

The initial protocol is:

| Variable | Meaning |
|---|---|
| `USER1` | protocol version |
| `USER2` | engine tic |
| `USER3` | task phase |
| `USER4` | core state: center, carried, dropped, returned |
| `USER5` | pickup counter |
| `USER6` | drop counter |
| `USER7` | score counter |
| `USER8` | valid-hit counter |
| `USER9` | death counter |
| `USER10` | respawn counter |
| `USER11` | core-timeout-return counter |
| `USER12` | task-success counter |
| `USER13` | target state |
| `USER14` | public home score |
| `USER15` | public away score |
| `USER16` | reserved; must remain zero in protocol v1 |
| `USER17` | privileged core X coordinate as fixed point |
| `USER18` | privileged core Y coordinate as fixed point |
| `USER19` | privileged target X coordinate as fixed point |
| `USER20` | privileged target Y coordinate as fixed point |

Python snapshots these variables before and after each decision. Positive
counter deltas produce typed `EpisodeEvent` objects; a delta less than zero,
greater than the documented per-step bound, or a protocol-version mismatch is
a hard error. Region transitions are derived from legal engine position only
inside the privileged event adapter and checked against `regions.yaml`.

Each event contains `episode_id`, engine tic, decision index, event type,
subject, optional region transition, and numeric value. Events are immutable,
JSON-serializable, ordered, and reset between episodes. Pickup, score, hit, and
progress rewards are computed from events in Python with per-episode caps;
reading the same counter snapshot twice cannot award twice.

## 7. Fairness Boundary and Environment API

Three types prevent accidental information leakage:

- `ActorObservation`: `84×84` grayscale first-person frame, health, ammunition
  or attack readiness, core possession, public scores, remaining time, and the
  previous macro action.
- `PrivilegedState`: exact positions, angles, current region, core location,
  target state, visibility, and task phase. Only Teachers, event generation,
  reward checks, and offline evaluation may consume it.
- `EpisodeEvent`: the audited transition record described above.

`SingleAgentTaskEnv.reset(seed, task)` returns an `ActorObservation` and reset
metadata. `step(action)` returns observation, scalar reward, `terminated`,
`truncated`, and typed events. Invalid actions, missing frames, event protocol
violations, and dead engine processes fail loudly. `close()` is idempotent, and
the engine is closed on all construction and stepping failures.

The environment maps the 13 approved macro actions to ViZDoom button vectors.
The mapping is fixed, unit-tested, and shared by Teachers and later policies.
No labels, depth, automap, region ID, target coordinate, or exact enemy state is
present in `ActorObservation` or accepted by a future policy forward method.

## 8. Teachers and Baselines

Teachers are deterministic finite-state controllers over `PrivilegedState`:

1. `FixedRouteTeacher`: follows a selected route to a requested region.
2. `ObjectiveFirstTeacher`: searches for the core, picks it up, and returns by
   the shortest safe route.
3. `AggressiveScriptTeacher`: aligns with visible targets, closes distance, and
   attacks only under a valid firing condition.
4. `DefensiveScriptTeacher`: holds a configured choke or home region and engages
   targets entering its defensive envelope.
5. `EvasiveReturnTeacher`: returns a carried core using the flank route and
   deterministic disengagement turns.

`RandomLegal` is included as a reproducible negative-control baseline but is not
called a Teacher and is not expected to pass capability gates.

FSM transitions and route selection are logged. Teachers never enter Actor
observations and are not presented as a strength upper bound. They exist to
validate action expressiveness, generate future demonstrations, and form a
clear scripted capability exam.

## 9. Split and Evaluation Protocol

A pure configuration generator maps a master seed to immutable train,
validation, and test manifests. The manifests contain scenario version, task,
spawn choice, core choice, target path, requested route or region, and episode
seed. Their hashes and seed sets must be pairwise disjoint.

Development may use train seeds. Teacher parameters and route tolerances may be
selected only on validation seeds. The accepted capability table is computed
once from the frozen test manifest by `scripts/evaluate_m1.py` and saved as JSON
and CSV. The evaluator records the Git commit, WAD hash, protocol version,
Teacher name, task, seed, outcome, decisions, and event counts.

No training process or `nohup` job is required in M1.

## 10. Testing and Failure Handling

Unit tests cover:

- deterministic WAD packing and manifest hashing;
- region membership, boundary policy, and graph paths;
- the 13 macro-action vectors and invalid actions;
- Actor observation shape/dtype/schema and privileged-field exclusion;
- event counter decoding, deduplication, reset, bounds, and JSON serialization;
- reward positive cases, negative cases, caps, and reset;
- Teacher FSM transitions and deterministic choices;
- split disjointness and reproducibility.

Integration tests cover:

- ACC compilation and reproducible scenario build when ACC is available;
- tracked WAD load, frame access, and natural timeout in headless ViZDoom;
- pickup, return, score, static hit, moving hit, and region-transition events;
- negative traces that must not emit pickup, score, or valid-hit events;
- every task's reset/step/termination contract;
- same-seed event-sequence replay;
- required MP4 recording and engine cleanup.

The M1 evaluator exits nonzero if any threshold is missed, any event mismatch is
observed, a test seed overlaps another split, or an expected artifact is absent.

## 11. Public Presentation

After the gate passes, the README gains a concise M1 section with:

- one arena image with labeled routes;
- the five capability rates and confidence intervals;
- one short Teacher montage;
- a compact example event trace;
- exact build, smoke, and evaluation commands;
- an explicit statement that no learned Bot or multiplayer result exists yet.

Raw M1 JSON/CSV results and their manifest are committed under
`reports/m1/`; transient videos and development logs remain ignored under
`artifacts/`.
The tracked showcase image is stored under `docs/assets/`.

## 12. References

- [ViZDoom custom environment guide](https://vizdoom.farama.org/environments/creating_custom/)
- [ViZDoom game variables](https://vizdoom.farama.org/main/api/python/enums/)
- [ViZDoom configuration files](https://vizdoom.farama.org/1.2.3/api/configurationFiles/)
- [ZDoom ACS reference](https://zdoom.org/wiki/ACS)

## 13. Self-Review

- No placeholders or unresolved choices remain.
- The design preserves the approved single-instance M1 boundary.
- Actor and privileged inputs are structurally separated.
- The arena has one geometry source despite task-specific map markers.
- Every accepted metric has a sample count, split rule, and machine-verifiable
  threshold.
- M2 learning and multiplayer interfaces are not invented in M1.
