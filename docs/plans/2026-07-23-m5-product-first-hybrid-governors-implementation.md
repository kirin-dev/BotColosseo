# M5 Product-First Hybrid Governors Implementation Plan

**Approved design:** `docs/superpowers/specs/2026-07-23-m5-product-first-hybrid-governors-design.md`

## Outcome

Ship auditable Defensive and Explorer hybrid policies that wrap the unchanged
learned Strong Base, use only public Actor inputs, retain task performance, and
produce real showcase evidence. The existing learned-style failures and frozen
legacy evaluators remain unchanged.

## Non-negotiable boundaries

- The behavioral path accepts `DuelActorObservation` only.
- No seed, case identity, learner side, coordinates, region labels, Teacher
  state, privileged carrier identity, opponent health, or Critic features.
- The Base checkpoint, scenario hash, and governor config are hash-bound.
- Every intervention is deterministic, finite, bounded, and telemetered.
- Existing Defensive and Explorer records/summaries stay frozen and are reused
  only as diagnostics.
- Defensive and Explorer are published as **hybrid governors**, not learned
  reward-shaped policies.

## Phase 1 — Pure governor domain

Add `src/botcolosseo/agents/style_governor.py`.

1. Define immutable, validated:
   - `PublicStyleContext`;
   - `GovernorDecision`;
   - `GovernorTelemetry`;
   - Defensive and Explorer config dataclasses.
2. Implement `DefensiveGovernor` states:
   - `base`, `guard`, `disengage`, `recover`;
   - score-rise Guard;
   - health-drop/low-health Disengage;
   - carrying and cooldown fallbacks.
3. Implement `ExplorerGovernor` states:
   - `base`, `route_commit`, `stall_recovery`;
   - `episode_counter % 3` route commitment on `has_core` rising edge;
   - bounded Upper, Lower, and Flank rhythms;
   - score/drop/health/duration recovery.
4. Test every state boundary, mirrored route signatures, deterministic replay,
   finite biases, legal overrides, consecutive limits, and reset behavior.

**Gate:** pure unit tests pass and public context has no privileged fields.

## Phase 2 — Strong Base hybrid policy

Add `src/botcolosseo/agents/hybrid_policy.py`.

1. Load a schema-1 Strong Base league checkpoint with exact SHA-256 and scenario
   checks, using the existing public Actor architecture.
2. Run the frozen CNN-GRU once per decision and expose only Base logits to the
   governor.
3. Apply finite bias by default; allow only explicitly bounded legal overrides.
4. Fall back to exact Base argmax on inactive, invalid, exhausted, or safety
   decisions.
5. Keep an append-only in-memory telemetry stream and provide an evaluation
   adapter whose privileged argument is immediately discarded.
6. Reset Actor and governor state without using the evaluation seed.

**Gate:** checkpoint/hash rejection, exact Base fallback, deterministic action
sequence, and no privileged policy input are covered by unit tests.

## Phase 3 — Frozen configs and identity

Add:

- `configs/m5/hybrid/defensive.yaml`;
- `configs/m5/hybrid/explorer.yaml`;
- a strict config loader and candidate identity builder.

Each candidate binds:

- Base checkpoint path and SHA-256;
- scenario hash;
- governor config hash;
- code revision;
- validation case-manifest hash;
- `test_cases_accessed: false`.

Only the small validation grid declared in the approved design may be compared.
No test split or showcase clip may select a candidate.

**Gate:** unknown fields, invalid thresholds, bad hashes, or test access fail
closed.

## Phase 4 — Product evaluation without evaluator drift

Add `src/botcolosseo/evaluation/hybrid.py` and
`scripts/evaluate_hybrid.py`.

1. Reuse `run_defensive_episode` and `run_explorer_episode` unchanged.
2. Pair each hybrid with the exact Strong Base on the frozen validation
   schedule.
3. Reuse the old summaries as legacy diagnostics, but compute separate hybrid
   hard gates:
   - 20/20 complete;
   - zero protocol inconsistencies;
   - retention `>= 0.85`;
   - every opponent retention `>= 0.75`;
   - nonzero bounded intervention;
   - required state/mode coverage;
   - no overwrite and no test access.
4. Emit episode JSONL, telemetry JSONL, summary JSON, hashes, and an auditable
   manifest in a fresh output directory.

**Gate:** mocked CLI/evaluation tests pass, then each real 20-episode CUDA smoke
passes all hybrid hard gates. A failed candidate stops before formal evaluation.

## Phase 5 — Formal evidence and showcase

For each passing smoke:

1. Run 200 paired validation episodes.
2. Report performance, per-opponent retention, state occupancy, intervention,
   fallback, action signatures, and unchanged legacy diagnostics.
3. Record one real MP4 for each hybrid.
4. Build the real four-policy 2-by-2 GIF.
5. Update public docs and release metadata with honest labels:
   - Strong Base — learned;
   - Aggressive — learned style;
   - Defensive — hybrid governor;
   - Explorer — hybrid governor.
6. Prepare the randomized label-blind study and require at least 10 responses,
   overall recognition `>= 60%`, and each style `>= 50%`.

Long real evaluations are launched with `nohup`, inspected at least twice, and
then allowed to continue in the background.

## Implementation order and verification

1. Pure governor types and tests.
2. Hybrid checkpoint policy and tests.
3. Strict YAML configs and identity tests.
4. Evaluation aggregation and CLI tests.
5. Fast lint/unit suite.
6. Real 20-episode Defensive smoke.
7. Real 20-episode Explorer smoke.
8. Only after both pass: formal evaluation and showcase integration.

At each phase, commit only the scoped implementation and its evidence. Never
rewrite or delete the preserved V1/V2 negative artifacts.
