# M5 Hybrid-Aware All-Style Difficulty

**Status:** Approved in conversation on 2026-07-23; written specification
awaiting owner review.

## 1. Outcome

The public product needs one honest `Style × Difficulty` matrix for:

- Strong Base;
- Aggressive learned style;
- Defensive hybrid governor;
- Explorer hybrid governor;
- Easy, Normal, and Hard.

The matrix contains exactly `4 policies × 3 difficulties × 100 validation
cases = 1,200` unique cells. It is an evaluation and packaging milestone, not
new training. No policy weights, governor parameters, difficulty profiles,
validation cases, or thresholds are tuned after observing the result.

This specification supersedes the learned-only 1,800-row pairwise-block design.
It preserves the frozen difficulty controller and the passed hybrid product
policies.

## 2. Evidence correction

The legacy combined audit required repeated Strong Base runs to have identical
episode outcomes. Current artifacts contradict that assumption:

- Aggressive difficulty Hard versus Defensive formal Hard: 51 of 100 common
  cases differ in at least one outcome field;
- Aggressive difficulty Hard versus Explorer formal Hard: 35 of 100 differ;
- Defensive formal Hard versus Explorer formal Hard: 44 of 100 differ.

All source identities remain fixed, but independent ViZDoom executions are not
episode-outcome deterministic. Requiring identical outcomes would reject valid
evidence for an engine property the product does not promise.

The hybrid-aware audit therefore requires exact artifact, case, seed, side,
opponent, scenario, policy, and protocol identity. It does not require outcomes
from separate engine executions to be byte-identical.

## 3. Frozen matrix and provenance

The matrix reuses immutable evidence by hash:

| Cells | Rows | Source |
|---|---:|---|
| Strong Base, Easy/Normal/Hard | 300 | Existing M5 difficulty formal ledger |
| Aggressive, Easy/Normal/Hard | 300 | Existing M5 difficulty formal ledger |
| Defensive, Hard | 100 | Passing Defensive hybrid formal ledger |
| Explorer, Hard | 100 | Passing Explorer Candidate C formal ledger |
| Defensive, Easy/Normal | 200 | New constrained hybrid evaluation |
| Explorer, Easy/Normal | 200 | New constrained hybrid evaluation |

Only 400 new formal episodes are required. Hard is reusable because the frozen
Hard profile `(reaction_delay=0, policy_update_interval=1)` is exactly the
native policy stream used by the formal hybrid evaluations.

The canonical Strong Base cells come from the existing difficulty ledger.
Hard retention recalculated against that canonical Base already satisfies the
unchanged product thresholds:

- Defensive aggregate `0.9817`, minimum opponent `0.9231`;
- Explorer aggregate `1.0052`, minimum opponent `0.9875`.

The source ledgers remain unchanged. A new matrix manifest references source
paths and SHA-256 values instead of copying or relabeling their rows.

## 4. Runtime composition

For the new Easy and Normal runs, `DifficultyPolicy` wraps the complete public
hybrid policy:

```text
public observation
        |
Strong Base CNN-GRU -> hybrid governor -> governed action
        |
frozen DifficultyPolicy -> executed action
```

This ordering makes Style and Difficulty independent product controls:

- the hybrid policy decides what behavior it wants;
- the difficulty controller limits how often that complete Bot can reconsider
  and when its selected action reaches the environment.

The controller is not inserted inside the Strong Base actor. Doing so would
let the governor run at a different cadence from learned policies, change the
meaning of the frozen controller, and invalidate cross-policy comparison.

Both layers receive only `DuelActorObservation`. Privileged state is available
only to the existing environment runner and offline evaluators.

## 5. Executed-action provenance

Easy and Normal must prove visible executed behavior, not merely a governor
intent that is later hidden by holding or delay.

The evaluation adapter observes the unchanged `DifficultyPolicy` and maintains
a fail-closed mirror of its action provenance:

1. On a policy-update decision, exactly one hybrid governor telemetry record is
   consumed and attached to the proposed action.
2. On a held decision, the previous proposal and provenance are reused.
3. The proposal and provenance enter a mirror FIFO with the frozen reaction
   delay.
4. The mirror's predicted action must equal the actual action emitted by
   `DifficultyPolicy`.
5. FIFO warm-up `IDLE` actions carry an explicit `warmup` source and cannot be
   counted as a style intervention.

Each environment decision records:

- policy, difficulty, case identity, and decision index;
- whether the underlying hybrid policy updated;
- proposed governed action;
- emitted action;
- source governor decision index;
- state, trigger, reason, route mode, and intervention flag;
- whether the row is FIFO warm-up.

The trace is evidence-only. It does not alter policy state or emitted actions.
For an episode with `D` decisions and update interval `u`, the underlying policy
must be called exactly `ceil(D / u)` times.

## 6. Frozen acceptance gates

### 6.1 Matrix integrity

- exactly 1,200 unique `(policy, difficulty, opponent, pair_index,
  learner_side)` cells;
- exactly 100 validation cases in every policy/difficulty cell;
- all source ledgers and summaries match their frozen SHA-256 values;
- no duplicate, missing, truncated, or protocol-inconsistent episode;
- scenario, cases, difficulty config, policy artifacts, split, and selected
  case identities match;
- `test_cases_accessed: false` throughout.

### 6.2 Difficulty behavior

For every policy:

- aggregate performance is Easy ≤ Normal ≤ Hard using the frozen `0.03`
  adjacent-tier tolerance;
- at least four of five opponents are monotonic;
- Easy objective rate is at least 70% of Hard;
- Normal objective rate is at least 85% of Hard.

No tier may borrow performance or style evidence from another tier.

### 6.3 Hybrid skill retention

At every tier, relative to the canonical Strong Base cell at the same
difficulty:

- aggregate Skill Retention is at least `0.85`;
- every opponent retention is at least `0.75`.

Zero-valued Base denominators fail closed rather than being smoothed.

### 6.4 Hybrid style mechanism

The existing Hard source must retain its already-passing hybrid product
summary and exact governor config hash.

For Defensive Easy and Normal, separately:

- governor telemetry and executed-action provenance are complete;
- intervention count is nonzero;
- `guard`, `disengage`, and `recover` are all exercised;
- non-intervened proposals equal the exact Strong Base proposal;
- consecutive proposed interventions do not exceed the governor limit;
- consecutive executed intervention influence does not exceed
  `governor_limit × policy_update_interval`.

For Explorer Easy and Normal, separately:

- governor telemetry and executed-action provenance are complete;
- intervention count is nonzero;
- `upper`, `lower`, and `flank` are all exercised;
- non-intervened proposals equal the exact Strong Base proposal;
- proposed and executed intervention bounds both hold;
- executed-action route signature distance is at least the frozen `0.05`.

The failed legacy learned-policy diagnostics remain visible and are not
converted into product passes.

## 7. Artifacts and audit

The new evaluator writes one resumable directory per hybrid style containing:

- `run.json`;
- `episodes.jsonl` for Easy/Normal;
- `telemetry.jsonl` for governor decisions;
- `execution-trace.jsonl` for every environment decision;
- `summary.json`;
- `manifest.json`.

The combined audit writes:

- `reports/m5/difficulty/hybrid-all-style-summary.json`;
- `docs/assets/showcase/m5-hybrid-all-style-difficulty.png`;
- a hash-bound M6 metric payload.

The audit uses a new schema and stage name. It does not weaken or overwrite the
legacy learned-only audit. Failed smokes or formal runs remain preserved.

## 8. Execution sequence

1. Implement trace composition, hybrid Easy/Normal evaluation, matrix audit,
   plot, tests, and runbook.
2. Run CPU/unit/static verification.
3. Run a small real ViZDoom smoke for both styles.
4. If both smokes are protocol-clean, launch one detached formal pipeline for
   the 400 new episodes.
5. Inspect the process and progress ledger at least twice.
6. After completion, run the combined audit, plot, M6 export, documentation
   update, release rebuild, and final repository audit.

The formal run is resumable and refuses identity drift. No user-run command is
needed unless the server environment prevents Codex from starting the detached
process.

## 9. Alternatives rejected

### Keep the 1,800-row pairwise layout

This repeats Strong Base 900 times and depends on a cross-run determinism check
already contradicted by real artifacts. It costs more and produces a less
direct product matrix.

### Apply difficulty inside the hybrid policy

Constraining only the Base actor while the governor runs at full cadence makes
Difficulty mean something different for hybrid and learned policies. It also
requires invasive policy changes and breaks the frozen controller boundary.

### Omit all-style difficulty

The existing showcase remains useful, but M5/M6 could not honestly claim the
planned independent Style × Difficulty product axis.

## 10. Non-goals

- no retraining or reward changes;
- no new governor candidate grid;
- no difficulty-specific thresholds or profiles;
- no test-split evaluation;
- no relabeling of failed learned Defensive/Explorer routes;
- no claim that independent ViZDoom runs are episode-outcome deterministic.
