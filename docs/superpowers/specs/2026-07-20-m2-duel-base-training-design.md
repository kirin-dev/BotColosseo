# Milestone 2 Design: Synchronous Duel and Base Training

**Status:** Approved route elaboration. This design narrows the already approved
M2 section of `Plan.md`; it does not change the product story, Actor fairness
boundary, 13-action contract, or M3 Strong Base thresholds.

## 1. Outcome and non-goals

M2 delivers the first learned visual Bot in a real two-player Crystal Run match:

1. a bounded, process-isolated `SynchronousDuelEnv`;
2. auditable Teacher demonstrations using only train cases;
3. a recurrent visual BC checkpoint;
4. a lightweight recurrent PPO implementation initialized from BC;
5. a frozen paired evaluation showing PPO is materially better than BC and a
   random legal policy.

M2 does not claim a Strong Base, PFSP robustness, style shaping, difficulty
control, or human evaluation. Historical opponents and the final Strong Base
gate remain M3.

## 2. Why the duel runtime uses two worker processes

ViZDoom multiplayer is a host/join network game, not two independent local
simulations. The host launches with `-host 2`; the opponent joins an explicit
loopback port. Each worker owns exactly one `DoomGame` instance.

Both games run in synchronous `PLAYER` mode. For each tic of a macro action, the
coordinator sends both actions before either result is awaited, then collects
both results as an explicit barrier. Each worker performs exactly:

```text
set_action(action)
advance_action(1, update_state=is_last_tic)
```

The coordinator repeats that barrier for the same four engine tics and updates
the rendered state only on the final tic. This is required because ViZDoom's multiplayer
FAQ states that synchronous players must each advance one frame before the
server proceeds; multiplayer frameskip cannot be treated as one independent
`make_action` call. Process isolation also contains engine crashes, allows hard
timeouts, and prevents one blocked instance from hanging the learner.

Primary references:

- https://vizdoom.farama.org/faq/#having-multiple-agents-in-one-game-multiplayer-issues
- https://vizdoom.farama.org/main/api/python/doom_game/#vizdoom.DoomGame.advance_action
- https://github.com/Farama-Foundation/ViZDoom/tree/1.3.0/examples/python

## 3. Crystal Run duel scenario

The source-built WAD gains `MAP07`, reusing the reviewed arena geometry and
adding two deterministic deathmatch starts. The ACS protocol is extended in a backwards-compatible
versioned block for:

- current carrier (`none`, host, opponent);
- host and opponent scores;
- pickup, drop, score, death, and respawn counters per side; valid hits are
  decoded from each player's native ViZDoom `HITCOUNT` delta;
- round state and terminal winner;
- core coordinates and deterministic spawn index.

Shared values may be read by both processes, but never enter the Actor beyond
the approved public scalars. Local health, armor, ammo, own-score, opponent-score,
own possession, last action, and the 84×84 grayscale frame form the duel Actor
observation. Player coordinates, angles, opponent position, core position,
region IDs, and ACS phase remain privileged.

The round lasts at most 2,100 tics (60 seconds at 35 Hz) and ends early when a
side reaches three scores. Death triggers a fixed respawn delay; a dropped core
returns to a deterministic central candidate after a fixed timeout. All combat
uses the existing hitscan weapon and no auto-aim.

## 4. Environment contract and synchronization evidence

`SynchronousDuelEnv.reset(seed, case)` returns a `DuelObservation` for each side
and immutable reset metadata. `step(host_action, opponent_action)` returns a
`DuelStep` containing both legal observations, side-specific task rewards,
shared immutable events, termination/truncation, scores, decision index, and
engine tic.

The coordinator enforces:

- unique loopback port per live duel;
- bounded host/join initialization;
- both commands queued before either result is read;
- host-authoritative ACS time advances exactly four tics per decision; peer
  replication lag is recorded and may not exceed two tics around respawn;
- equal decision counts and shared protocol counters;
- bounded reset, step, and close;
- forced worker termination only after graceful close fails;
- no live child processes or bound ports after close.

The real synchronization gate runs at least 10,000 consecutive decisions across
resets and deaths with zero tic mismatch, timeout, protocol inconsistency, or
orphan worker. A short MP4 records both perspectives side by side.

## 5. Script opponents and duel Teacher

The five existing policies become duel-capable finite-state controllers:

- `RandomLegal` samples only the fixed 13 actions;
- `FixedRoute` follows a declared route to the core and home;
- `ObjectiveFirst` prioritizes pickup and scoring;
- `AggressiveScript` intercepts a visible or recently seen carrier;
- `DefensiveScript` protects its base and disengages when appropriate.

`DuelTeacher` selects among objective, engage, evade, and recover states from
`DuelPrivilegedState`. Teacher state is never serialized into Actor input.
Teacher and opponent names, FSM state transitions, actions, and public events
are recorded for audit.

## 6. Demonstration dataset

Demonstrations are deterministic compressed NPZ shards plus a JSON manifest.
The deterministic writer produces identical NPZ bytes for identical arrays.
Because multiplayer sprite interpolation may vary cosmetically across process
restarts, each shard also records a frame-excluding trajectory hash that proves
the scalars, actions, labels, and boundaries reproduce from the same cases;
the full shard hash always validates the exact stored visual artifact.
Each transition contains only:

- `frame`: uint8 `[84, 84]`;
- normalized legal scalars;
- previous macro action;
- Teacher action label;
- episode/sequence boundary and valid mask;
- task/opponent identifiers and train seed for provenance.

Privileged state may be used online by the Teacher but is forbidden from shard
schemas. A schema test recursively rejects privileged field names. Generation
uses only frozen train cases. Validation uses validation cases; test cases are
not opened by data generation or checkpoint selection.

The initial target is 100,000 train and 20,000 validation transitions, balanced
across objective phases and the five opponent types by bounded downsampling.
Full shards are ignored because they are reproducible and large; the repository
tracks configs, manifests, per-shard hashes, aggregate action/phase counts, and
a tiny schema sample.

## 7. Recurrent visual model and BC

The shared Actor follows `Plan.md`:

```text
84x84 grayscale -> 3-layer CNN -> 256-d feature
legal scalars + previous action -> small MLP
concatenate -> GRU(256) -> categorical 13-action policy
```

The Critic has a separate privileged encoder concatenated only after the Actor
GRU output. Tests prove policy logits are unchanged when privileged tensors are
changed or omitted.

BC trains sequence chunks with hidden-state resets at episode boundaries,
cross-entropy loss, gradient clipping, deterministic validation, and atomic
checkpoint/resume. Selection uses validation action loss and closed-loop
validation objective rate, not train accuracy. The repository publishes the
best pure-BC checkpoint as the M2 baseline.

## 8. Recurrent PPO

PPO starts from the selected BC Actor and uses:

- GAE and bootstrapping that distinguishes terminal from timeout;
- clipped policy and value objectives;
- entropy bonus, advantage normalization, and gradient clipping;
- fixed-length recurrent sequence minibatches with burn-in masks;
- asymmetric Critic inputs that never enter policy logits;
- NaN/Inf guards, optimizer/scheduler state, RNG state, and atomic resume;
- per-component reward and event logging.

The M2 curriculum is fixed before test evaluation:

1. full objective against `RandomLegal` and `FixedRoute`;
2. add `ObjectiveFirst`;
3. add `AggressiveScript` and `DefensiveScript` with a uniform mixture.

Early progress/pickup/hit shaping decays according to environment steps. Score,
win, death, and stall terms remain. All bounded shaping reuses event preconditions
and caps; no visual-policy reward is computed from future information.

One A100 runs the learner and rollout batches. The second GPU is optional for
evaluation; M2 does not introduce a distributed learner.

## 9. Frozen M2 evaluation and gate

Before any official run, commit train/validation/test duel manifests and the
evaluation config. The official test uses the same paired seeds and side swaps
for PPO, BC, and `RandomLegal` against the five fixed script opponents. Each
policy receives 50 seed-pairs per opponent (500 games after side swaps), with
win, draw, objective completion, score difference, and Wilson/bootstrap 95%
intervals reported.

M2 passes only if all engineering gates pass and PPO meets every performance
condition:

- PPO average win rate exceeds BC by at least 10 percentage points;
- PPO average win rate exceeds `RandomLegal` by at least 20 percentage points;
- PPO objective completion exceeds BC by at least 10 percentage points;
- paired bootstrap 95% lower confidence bound for PPO minus BC mean score
  difference is greater than zero;
- no opponent has PPO win rate more than 5 points below BC;
- zero protocol, fairness-schema, synchronization, or artifact inconsistencies.

These are M2 learning-progress gates, not the M3 Strong Base thresholds. If
validation shows a gate is structurally inappropriate, it may be revised once
with a committed rationale before test is opened. Test results never tune it.

## 10. Failure and experiment boundary

Every deterministic component is implemented and tested before long training.
Short CPU/GPU smoke runs must prove forward/backward, checkpoint resume, rollout,
and evaluation end-to-end.

If the first meaningful BC or PPO run requires a manual background process, the
exact `nohup` command, log path, expected checkpoints, progress checks, success
criteria, and resume command are appended to `script.md`. Work then stops for
the user to launch it, matching the agreed experiment handoff rule.

## 11. Public packaging

M2 publishes:

- synchronization audit JSON and dual-perspective video;
- demonstration manifest and action/phase distribution;
- BC and PPO training curves with config hashes;
- frozen PPO-vs-BC-vs-random table and raw episode CSV;
- exact checkpoint/evaluation commands;
- an explicit statement that M3 PFSP/Strong Base and all styles remain pending.
