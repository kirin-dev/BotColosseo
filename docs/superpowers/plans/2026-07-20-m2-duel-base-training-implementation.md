# Milestone 2 Synchronous Duel and Base Training Implementation Plan

> Execute task by task with strict TDD. Commit deterministic code and frozen
> configs before opening test evidence. Do not launch a manual background
> experiment until its exact handoff is recorded in `script.md`.

**Goal:** Deliver a real synchronized Crystal Run 1v1 environment, reproducible
Teacher demonstrations, recurrent visual BC and PPO, and frozen evidence that
PPO materially outperforms BC and random.

**Architecture:** Two spawned worker processes each own one synchronous ViZDoom
instance and communicate with a bounded coordinator. A backwards-compatible
MAP07 ACS protocol drives shared duel events. Reproducible NPZ demonstration
shards feed one CNN-GRU Actor; BC initializes recurrent PPO with a separate
privileged Critic path. Frozen paired side-swapped evaluation decides M2.

**Environment:** Python 3.10, ViZDoom 1.3.0, PyTorch 2.6, NumPy, OpenCV,
Matplotlib, FFmpeg, pytest, Ruff, ACC 1.60.

**Design:** `docs/superpowers/specs/2026-07-20-m2-duel-base-training-design.md`

## Global rules

- Use `/home/wencong/miniconda3/envs/botcolosseo/bin/python` explicitly.
- Preserve the 13 macro actions and `ActorObservation` fairness boundary.
- In synchronous multiplayer, advance exactly one tic per player command; do
  not call multiplayer `make_action(..., frame_skip)`.
- Use bounded IPC waits and close workers in `finally`.
- Never tune using `configs/m2/test.json` or official M2 episode rows.
- Generated datasets, training runs, and checkpoints stay ignored; tracked
  manifests and published selected checkpoints are copied deliberately.
- A failing test precedes every implementation or bug fix.

---

### Task 1: Extend Crystal Run with a versioned duel protocol

**Files:**

- Modify: `assets/scenarios/crystal_run/src/map.udmf`
- Modify: `assets/scenarios/crystal_run/src/crystal_run.acs`
- Modify: `assets/scenarios/crystal_run/crystal_run.cfg`
- Modify: `src/botcolosseo/scenarios/build.py`
- Create: `src/botcolosseo/envs/duel_protocol.py`
- Create: `tests/unit/test_duel_protocol.py`
- Modify: `tests/unit/test_scenario_build.py`
- Modify: `tests/integration/test_scenario_build.py`

1. Write failing tests for a MAP07 lump, two player starts, protocol version 2,
   side-specific monotonic counters, carrier/winner ranges, and decoder reset,
   decrement, jump, and reserved-field failures.
2. Run the focused tests and retain the RED output.
3. Add MAP07 by reusing arena geometry and adding deterministic player 1/player
   2 starts. Extend ACS without changing MAP01–MAP06 behavior.
4. Implement frozen `DuelProtocolSnapshot`, `DuelEvent`, and decoder types. Keep
   shared protocol fields out of `ActorObservation`.
5. Rebuild with ACC, require deterministic bytes, load all seven maps in real
   ViZDoom, and run all existing M1 tests.
6. Commit: `feat: add Crystal Run duel protocol`.

Gate: byte-identical rebuild, MAP01–MAP07 load, M1 evidence semantics remain
valid, and duel protocol negative tests pass.

---

### Task 2: Define duel observations, privileged state, rewards, and splits

**Files:**

- Create: `src/botcolosseo/envs/duel_types.py`
- Create: `src/botcolosseo/envs/duel_rewards.py`
- Create: `src/botcolosseo/scenarios/duel_splits.py`
- Create: `configs/m2/train.json`
- Create: `configs/m2/validation.json`
- Create: `configs/m2/test.json`
- Create: `configs/m2/reward.yaml`
- Create: `tests/unit/test_duel_types.py`
- Create: `tests/unit/test_duel_rewards.py`
- Create: `tests/unit/test_duel_splits.py`

1. Write schema leak tests for legal frame/scalars/previous action and separate
   privileged positions, carrier, regions, and FSM state.
2. Test zero-sum score/win/death terms and bounded pickup/progress/hit/stall
   shaping, including repeated-event and oscillation counterexamples.
3. Test reproducible disjoint signed-32-bit-safe duel cases, explicit side swap,
   opponent balance, and exactly 50 test seed-pairs per opponent.
4. Implement the minimal frozen dataclasses and ledgers.
5. Generate all manifests once with master seed `20260720`; verify hashes and
   commit before any official evaluation.
6. Commit: `feat: freeze M2 duel contracts and splits`.

Gate: no privileged field can enter the Actor schema, rewards are bounded, and
train/validation/test seeds and side swaps are disjoint and balanced.

---

### Task 3: Implement bounded multiplayer worker IPC

**Files:**

- Create: `src/botcolosseo/envs/duel_worker.py`
- Create: `src/botcolosseo/envs/ipc.py`
- Create: `tests/unit/test_duel_worker.py`
- Create: `tests/unit/test_ipc.py`

1. Write fake-engine tests for `INIT`, `RESET`, `STEP`, `RESPAWN`, and `CLOSE`
   commands, stable request IDs, invalid ordering, timeout, remote exception,
   and forced shutdown.
2. Implement a spawned worker that creates/uses/closes `DoomGame` only inside
   the child. Never send frames through an unbounded queue.
3. Implement explicit host args (`-host 2`, loopback port, no autoaim, fixed
   respawn rules) and opponent join args. Sanitize names and port ranges.
4. For each macro decision, the coordinator barriers both workers after every
   one-tic `set_action/advance_action`; update state only on the final tic.
5. Prove fake workers receive both commands before the coordinator awaits
   results and that all child processes terminate after injected failures.
6. Commit: `feat: add bounded duel worker runtime`.

Gate: no unbounded blocking or orphan process is possible in unit tests.

---

### Task 4: Implement and smoke the real SynchronousDuelEnv

**Files:**

- Create: `src/botcolosseo/envs/synchronous_duel.py`
- Create: `src/botcolosseo/cli/smoke_duel.py`
- Create: `scripts/smoke_duel.py`
- Create: `tests/unit/test_synchronous_duel.py`
- Create: `tests/integration/test_synchronous_duel.py`

1. Write fake-worker tests for reset/step, equal tic enforcement, both-side
   observations/rewards, termination versus timeout, respawn, and cleanup.
2. Implement `SynchronousDuelEnv` with a loopback port reservation, bounded
   host/join startup, simultaneous step dispatch, shared event decoding, and
   typed results.
3. Write a real test that starts host and opponent, verifies multiplayer state,
   takes 100 paired decisions, asserts equal tics after every decision, resets,
   and closes without children.
4. Add a CLI that emits sorted JSON with scenario hash, port, decisions, engine
   tics, score, event counts, mismatch count, worker errors, and cleanup status.
5. Run the real smoke three times to expose port/startup races.
6. Commit: `feat: add synchronized Crystal Run duel environment`.

Gate: three consecutive real host/join runs pass with no mismatch, timeout, or
orphan process.

---

### Task 5: Add duel Teachers and the 10,000-decision synchronization gate

**Files:**

- Create: `src/botcolosseo/agents/duel_teachers.py`
- Create: `src/botcolosseo/evaluation/sync_audit.py`
- Create: `src/botcolosseo/cli/audit_duel_sync.py`
- Create: `scripts/audit_duel_sync.py`
- Create: `tests/unit/test_duel_teachers.py`
- Create: `tests/unit/test_sync_audit.py`
- Create: `tests/integration/test_duel_events.py`
- Produce: `reports/m2/sync-audit.json`
- Produce: `docs/assets/m2-sync-duel.mp4`

1. Write FSM tests for objective, intercept, evade, recover, defend, and random
   policies, including deterministic reset and no-action-leak cases.
2. Implement duel-capable controllers using only `DuelPrivilegedState`.
3. Add real positive and negative event tests for pickup/drop/score/hit/death/
   respawn and shooting without a valid target.
4. Implement an audit runner spanning deaths and resets. Record both grayscale
   perspectives side by side using only public overlay fields.
5. Run 10,000 consecutive decisions synchronously. Verify JSON counts, zero
   mismatch/errors, video duration/size, and no child processes.
6. Commit code, then evidence: `test: pass the M2 synchronization gate`.

Gate: exactly 10,000 audited decisions with zero mismatch, protocol error,
worker timeout, or orphan process.

---

### Task 6: Implement reproducible demonstration shards

**Files:**

- Add ignore: `/data/generated/`
- Create: `src/botcolosseo/data/schema.py`
- Create: `src/botcolosseo/data/demonstrations.py`
- Create: `src/botcolosseo/cli/generate_demonstrations.py`
- Create: `scripts/generate_demonstrations.py`
- Create: `configs/m2/demonstrations.yaml`
- Create: `tests/unit/test_demonstration_schema.py`
- Create: `tests/unit/test_demonstrations.py`
- Create: `tests/integration/test_demonstration_generation.py`
- Produce: `reports/m2/demonstrations-manifest.json`
- Produce: `docs/assets/m2-demonstration-distribution.png`

1. Write failing schema tests that recursively reject privileged keys and verify
   dtypes/shapes, episode boundaries, masks, opponent/task IDs, and hashes.
2. Implement bounded in-memory shard buffers and atomic compressed NPZ writes.
3. Generate a 200-transition real smoke shard and prove deterministic manifest
   hashes from the same cases.
4. Add balanced action/phase statistics and a deterministic distribution plot.
5. Generate the configured 100k train/20k validation dataset synchronously if
   it fits the foreground timeout. Otherwise write the exact background handoff
   to `script.md` and stop.
6. Commit reproducibility code/config and the generated manifest, never the full
   large shards.

Gate: schema leak count 0, requested transition counts exact, all shard hashes
valid, and no test case is accessed.

---

### Task 7: Implement the fair CNN-GRU Actor and asymmetric Critic

**Files:**

- Create: `src/botcolosseo/agents/model.py`
- Create: `src/botcolosseo/agents/checkpoint.py`
- Create: `configs/m2/model.yaml`
- Create: `tests/unit/test_model.py`
- Create: `tests/unit/test_checkpoint.py`

1. Test output/hidden shapes, sequence masks, deterministic inference, CPU/CUDA
   parity tolerance, and invalid inputs.
2. Prove policy logits are identical when privileged inputs change, while value
   predictions may change. Trace the exported Actor without a Critic input.
3. Implement the 3-layer CNN, scalar/previous-action encoder, GRU(256), policy
   head, and separate privileged value encoder.
4. Implement atomic versioned checkpoints containing model/optimizer/scheduler,
   config and scenario hashes, counters, and RNG states.
5. Prove save/resume produces the same next optimization update.
6. Commit: `feat: add recurrent visual actor critic`.

Gate: model forward/backward and exact resume tests pass on CPU and one A100
smoke; no privileged input influences policy logits.

---

### Task 8: Implement and smoke behavioral cloning

**Files:**

- Create: `src/botcolosseo/training/bc.py`
- Create: `src/botcolosseo/cli/train_bc.py`
- Create: `scripts/train_bc.py`
- Create: `configs/m2/bc.yaml`
- Create: `tests/unit/test_bc.py`
- Create: `tests/integration/test_bc_smoke.py`

1. Write tests for recurrent chunking, boundary masks, cross-entropy, validation,
   gradient clipping, deterministic loaders, best-checkpoint selection, and
   resume equivalence.
2. Implement the smallest trainer satisfying those tests with JSONL metrics and
   atomic checkpoints.
3. Run a tiny real dataset overfit test: loss must fall and action accuracy must
   rise over 50 updates.
4. Run a 1,000-update single-A100 smoke including interruption/resume and a
   closed-loop validation episode.
5. If the full BC run is long, append its exact `nohup`/log/checkpoint/validation
   commands to `script.md` and stop. Otherwise run it in foreground.
6. Commit: `feat: add recurrent behavioral cloning`.

Gate: smoke overfits, resume is equivalent, and a selected pure-BC checkpoint
has complete provenance.

---

### Task 9: Implement PPO and GAE math under unit tests

**Files:**

- Create: `src/botcolosseo/training/gae.py`
- Create: `src/botcolosseo/training/ppo.py`
- Create: `src/botcolosseo/training/rollout.py`
- Create: `configs/m2/ppo.yaml`
- Create: `tests/unit/test_gae.py`
- Create: `tests/unit/test_ppo.py`
- Create: `tests/unit/test_rollout.py`

1. Test hand-computed GAE for terminal versus timeout, recurrent masks, advantage
   normalization, clipped policy/value losses, entropy sign, and minibatches.
2. Add NaN/Inf and excessive-KL abort tests before implementation.
3. Implement typed rollout buffers and sequence minibatches with burn-in.
4. Implement PPO loss/update metrics without environment or CLI coupling.
5. Prove a deterministic synthetic batch lowers the expected objective and
   checkpoint resume repeats the next update.
6. Commit: `feat: add recurrent PPO core`.

Gate: all hand-computed numerical tests and safety guards pass.

---

### Task 10: Integrate duel rollouts, curriculum, and PPO smoke training

**Files:**

- Create: `src/botcolosseo/training/duel_rollout.py`
- Create: `src/botcolosseo/training/curriculum.py`
- Create: `src/botcolosseo/cli/train_ppo.py`
- Create: `scripts/train_ppo.py`
- Create: `tests/unit/test_curriculum.py`
- Create: `tests/integration/test_ppo_smoke.py`

1. Test the fixed opponent curriculum, shaping decay, train-only case access,
   hidden-state resets, opponent side swaps, and rollout accounting.
2. Integrate the BC checkpoint, duel env, Actor action sampling, asymmetric
   values, and recurrent PPO update.
3. Run 2,000 environment steps on one A100 and require finite metrics, at least
   one completed round, checkpoint save/resume, and cleanup.
4. Run a bounded 20k-step validation pilot against RandomLegal. Adjust only
   validation-visible optimizer/curriculum parameters with recorded rationale.
5. Freeze the selected config before official test evaluation.
6. Commit: `feat: integrate PPO duel training`.

Gate: end-to-end GPU smoke is finite, resumable, event-accounted, and leaves no
workers or GPU processes.

---

### Task 11: Implement frozen paired M2 evaluation

**Files:**

- Create: `src/botcolosseo/evaluation/m2.py`
- Create: `src/botcolosseo/cli/evaluate_m2.py`
- Create: `scripts/evaluate_m2.py`
- Create: `configs/m2/evaluation.yaml`
- Create: `tests/unit/test_m2_evaluation.py`
- Create: `tests/integration/test_m2_evaluation_smoke.py`

1. Test side-swapped paired scheduling, Wilson intervals, paired bootstrap with
   a fixed RNG, per-opponent tables, protocol counts, and every gate condition.
2. Make partial/development summaries visibly unofficial and unable to pass.
3. Implement atomic episode CSV, summary JSON, manifest hashes, checkpoint
   hashes, config hashes, and Git provenance.
4. Run a small real evaluation using random and script policies only; prove row
   counts, replay ordering, and worker cleanup.
5. Commit evaluator/config before opening official M2 test rows.
6. Commit: `feat: freeze M2 paired evaluator`.

Gate: fake and real smoke evidence cannot pass unless complete, official,
side-balanced, protocol-clean, and statistically above all frozen thresholds.

---

### Task 12: Run the meaningful BC and PPO experiments

**Files produced locally:**

- `runs/m2/bc/**`
- `runs/m2/ppo/**`
- `checkpoints/m2/**`

**Tracked after selection:**

- `reports/m2/bc-training-summary.json`
- `reports/m2/ppo-training-summary.json`
- selected compact checkpoints under `artifacts/m2/` only if repository size is
  acceptable; otherwise publish hashes and release instructions.

1. Audit both configs, dataset hashes, scenario hash, GPU visibility, disk, and
   absence of competing processes.
2. If either run cannot safely finish in the foreground, append exact launch,
   monitoring, completion, failure, and resume commands to `script.md`; then
   stop for the user to launch it.
3. After completion, validate logs, finite metrics, checkpoint readability,
   config hashes, and closed-loop validation—not merely a completion string.
4. Select checkpoints using validation only and record why each was selected.
5. Commit only compact summaries/provenance: `results: record M2 training runs`.

Gate: selected BC/PPO checkpoints are readable, reproducible, and chosen without
test access.

---

### Task 13: Run and record the official M2 gate

**Files:**

- Produce: `reports/m2/episodes.csv`
- Produce: `reports/m2/summary.json`
- Produce: `reports/m2/manifest.json`

1. Run the complete paired side-swapped test for PPO, BC, and random against all
   five frozen script opponents. Do not tune after seeing rows.
2. Verify exact expected row counts, seed/side balance, checkpoint and config
   hashes, no duplicates, zero sync/protocol inconsistencies, and all confidence
   outputs.
3. Require every gate in the M2 design. A failed gate is a real result; diagnose
   from validation/training evidence before authorizing any new experiment.
4. Commit passing evidence: `results: record Milestone 2 learning gate`.

Gate: official summary is true and PPO clears all absolute, paired, worst-case,
and integrity conditions.

---

### Task 14: Package and audit M2

**Files:**

- Create: `docs/milestones/m2.md`
- Create: `docs/assets/m2-training-curves.png`
- Create: `docs/assets/m2-policy-comparison.mp4`
- Modify: `README.md`
- Modify: `Plan.md`

1. Generate plots only from tracked summaries and a comparison video only from
   selected checkpoints on frozen cases.
2. Add exact runbooks, raw evidence links, limitations, and a clear statement
   that M3 Strong Base/PFSP and styles are pending.
3. Re-run environment check, deterministic WAD build, pip check, Ruff, full
   pytest, real duel smoke, sync artifact checks, checkpoint load, and official
   evidence audit.
4. Confirm `git diff --check`, no generated shards/runs/secrets are tracked, and
   the worktree is clean.
5. Commit: `docs: publish Milestone 2 evidence`.
6. Only after all checks pass, begin M3 design automatically.

Gate: M2 is reproducible, honestly scoped, visually demonstrable, and all
machine-verifiable evidence passes from a clean checkout.
