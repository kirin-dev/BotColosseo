# M5 Hybrid-Aware All-Style Difficulty Implementation Plan

**Approved design:** `docs/superpowers/specs/2026-07-23-m5-hybrid-all-style-difficulty-design.md`

## Outcome

Produce one audited 1,200-episode `Strong Base / Aggressive / Defensive /
Explorer × Easy / Normal / Hard` validation matrix. Reuse 800 immutable rows,
run only the 400 missing Defensive/Explorer Easy/Normal episodes, preserve the
public-observation boundary, and publish a real result without changing any
policy or threshold after observation.

## Non-negotiable boundaries

- `DifficultyPolicy` wraps the complete hybrid policy; its behavior and frozen
  profiles do not change.
- Policy code receives `DuelActorObservation` only.
- Existing episode, telemetry, summary, and manifest artifacts are immutable.
- Hard hybrid rows are reused by hash and are never relabeled as newly run.
- Cross-run outcomes are not required to match; source and case identities are.
- Formal evaluation is validation-only and must record
  `test_cases_accessed: false`.
- Failed product or difficulty gates remain published as failures.

## Phase 1 — Frozen source config and traced composition

Add a strict hybrid-difficulty source config binding:

- existing Base/Aggressive difficulty manifest;
- Defensive/Explorer Hard manifests and governor configs;
- difficulty config, cases, and scenario;
- expected SHA-256 values and zero test access.

Add a small traced adapter around the existing `DifficultyPolicy`:

1. build the existing `HybridStylePolicy` without exposing its internals;
2. mirror held-action and FIFO provenance without changing emitted actions;
3. require the mirror prediction to equal the real controller output;
4. bind every emitted action to governor state or explicit warm-up;
5. expose per-episode governor and execution traces.

**Verify:** unit tests cover Easy/Normal cadence, warm-up, exact action parity,
held provenance, reset isolation, intervention bounds, and public-only method
signatures.

## Phase 2 — Resumable Easy/Normal evaluator

Add one evaluator used for either Defensive or Explorer:

- only `easy` and `normal` are accepted;
- 100 frozen validation cases per tier;
- one style policy per run, for 200 rows total;
- existing defensive/explorer episode runners remain unchanged;
- ordered-prefix resume and exact run identity checks;
- append-only episode, governor telemetry, and execution trace ledgers;
- atomic summary and manifest with all ledger hashes.

The local style gate requires complete protocol-clean rows, complete trace
accounting, nonzero bounded intervention, exact Base fallback, and per-tier
Defensive state or Explorer executed-route coverage/signature.

**Verify:** synthetic aggregation tests cover pass, missing trace, action
parity drift, missing mode/state, signature failure, duplicate resume rows, and
identity drift. CLI preflight proves 200 expected episodes and no test access.

## Phase 3 — 1,200-row matrix audit

Add a new audit rather than modifying the legacy 1,800-row audit.

The audit:

1. validates every source path against the frozen config hash;
2. validates each source manifest's own episode/summary hash chain;
3. projects 600 Base/Aggressive rows, 200 reused hybrid Hard rows, and 400 new
   Easy/Normal rows into exactly 1,200 unique identities;
4. verifies 100 cases in every policy/tier cell and shared case identity;
5. computes common performance/objective cells;
6. applies monotonic, opponent monotonicity, objective capability, hybrid
   retention, and local style-mechanism gates;
7. emits a single immutable summary and fails closed on any mismatch.

**Verify:** tests prove acceptance of a complete synthetic matrix and rejection
of source hash drift, duplicate/missing cells, protocol drift, weak retention,
nonmonotonic performance, and test access. A fixture explicitly allows
different outcomes from repeated engine executions.

## Phase 4 — Product plot, M6 export, and runbook

Add:

- a four-policy Easy/Normal/Hard result card;
- hybrid-aware M6 metric export;
- a detached smoke/formal pipeline;
- exact build, progress, resume, audit, and plot commands in `script.md`.

Update `Plan.md` and README only after real evidence exists. Do not claim M5 or
Showcase-ready from synthetic tests or a smoke.

**Verify:** plot test requires a passing audited payload; export test verifies
all policy/config hashes and rejects failed gates.

## Phase 5 — Real execution and publication

1. Run Ruff and the full unit suite.
2. Run small real Defensive and Explorer Easy/Normal smokes.
3. Audit smoke ledgers and inspect process cleanup.
4. Launch the 400-episode formal pipeline detached on a free GPU.
5. Check PID/log/ledger progress at least twice.
6. After completion, run the combined audit and render the result card.
7. Update bilingual public status, M5/M6 evidence, release metadata, and
   reproducibility commands.
8. Run the final full test, artifact, hash, media, and Git audit; commit and
   push scoped evidence.

## Success criteria

- 1,200 unique validation episodes are hash-bound and protocol-clean.
- Strong Base/Aggressive reuse remains byte-identical.
- Defensive/Explorer Hard source hashes remain byte-identical.
- Only 400 new formal episodes are executed.
- Every frozen difficulty, retention, and style mechanism gate passes, or the
  actual failed gate is preserved and public claims remain partial.
- No ViZDoom worker remains after smoke completion.
- The branch is clean, pushed, and all public claims are backed by tracked
  evidence or a reproducible ignored artifact.
