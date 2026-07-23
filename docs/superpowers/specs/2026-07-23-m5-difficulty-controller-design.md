# M5 Difficulty Controller Design

**Status:** Approved through the project owner's delegated recommendation
authority on 2026-07-23.

## 1. Outcome

Bot Colosseo will expose `Easy`, `Normal`, and `Hard` without retraining a
policy or changing health, damage, observations, map state, or checkpoint
weights. Difficulty is a deterministic public-observation inference wrapper
around the same frozen style checkpoint.

The product goal is understandable challenge control, not an adaptive
difficulty algorithm. The wrapper must preserve the visible behavior identity
of Aggressive, Defensive, and Explorer while changing how quickly the Bot can
react.

## 2. Chosen interpretation

The original Plan described Normal as the native policy and Hard as a reduction
of inference restrictions. A native policy already has zero added delay and
updates on every environment decision, so delay and update cadence alone cannot
make a distinct Hard tier above it.

The frozen interpretation is therefore:

| Difficulty | Reaction delay | Policy update interval | Meaning |
|---|---:|---:|---|
| Easy | 2 decisions | 2 decisions | Delayed and slower to revise an action |
| Normal | 1 decision | 1 decision | Mildly humanized reaction lag |
| Hard | 0 decisions | 1 decision | Native frozen checkpoint |

This preserves the approved minimal mechanism and makes all three tiers
distinct. It also gives the strongest tier an honest meaning: Hard is the
unmodified learned Bot, not an oracle or a secretly stronger model.

## 3. Controller boundary

`DifficultyPolicy` wraps an existing recurrent checkpoint policy. Its public
surface is:

- `reset(seed)`: reset the checkpoint, action queue, held action, and step
  counter;
- `act(observation) -> MacroAction`: consume one public actor observation and
  return one legal macro action.

The wrapper never receives or resolves `DuelPrivilegedState`. On each decision:

1. If this is a policy-update decision, run the wrapped checkpoint on the
   current public observation and update the held action.
2. Otherwise, reuse the held legal action without advancing the recurrent
   checkpoint.
3. Push the chosen action into a FIFO of length `reaction_delay`.
4. Return the oldest delayed action, or `IDLE` while the FIFO is warming up.

Not advancing the model on held-action decisions makes `policy_update_interval`
a real perception/decision limitation instead of merely delaying an already
computed response. Every emitted value is validated as a `MacroAction`.

## 4. Configuration and identity

`configs/difficulty.yaml` is schema-versioned and contains exactly the three
frozen profiles. Values are nonnegative integers; interval must be positive;
Hard must be `(delay=0, interval=1)`; challenge restrictions must be ordered
Easy ≥ Normal ≥ Hard.

Evaluation and showcase manifests record:

- difficulty profile and config SHA-256;
- policy checkpoint SHA-256;
- scenario and case-manifest identities;
- validation split;
- `test_cases_accessed: false`.

No silent CLI overrides are allowed for formal evidence. Development smokes may
use a separate development config and cannot be published as the frozen gate.

## 5. Validation calibration

The calibration matrix initially uses every style checkpoint that has passed
its own style gate. Strong Base is always included as an integrity anchor.
Failed or pending style candidates may be inspected diagnostically but cannot
support the M5 claim.

Each policy × difficulty cell uses the same paired validation cases, opponents,
seeds, sides, episode limit, and public observation boundary. The evaluator
reports:

- win rate, objective rate, score difference, and combined performance;
- per-opponent performance;
- protocol inconsistencies and environment retries;
- style-specific metrics when the style has a passed evaluator;
- checkpoint, scenario, config, case, ledger, and summary hashes.

The frozen gate requires:

1. complete paired schedule and zero protocol inconsistencies;
2. aggregate performance approximately monotonic from Easy to Hard, allowing at
   most a 0.03 absolute sampling tolerance between adjacent tiers;
3. per-opponent monotonicity for at least four of five opponents;
4. Easy objective rate at least 70% of Hard and Normal at least 85% of Hard;
5. no style-signature direction reversal relative to Strong Base at the same
   difficulty;
6. zero test-case access.

The tolerance acknowledges finite validation samples without permitting a
materially inverted difficulty curve.

## 6. Anti-hacking and failure behavior

- Reset clears all delayed actions, so actions cannot leak across episodes.
- FIFO warm-up emits `IDLE`, never a random or previous-episode action.
- Held actions remain legal enum values.
- The wrapper cannot read privileged state, scores outside the public actor
  schema, opponent coordinates, or future observations.
- Invalid config, policy action, identity, duplicate ledger row, or resume
  mismatch fails closed.
- A non-monotonic validation result is preserved as a failed experiment; the
  thresholds are not weakened post hoc.

## 7. Engineering and product deliverables

- `DifficultyPolicy` plus focused unit tests;
- frozen `configs/difficulty.yaml`;
- resumable paired validation evaluator and evidence audit;
- one-command validation pipeline documented in `script.md`;
- showcase selector capable of loading one style checkpoint at any difficulty;
- README language that identifies Hard as native and Easy/Normal as constrained
  variants.

The first implementation does not include aim noise, action temperature,
dynamic player modeling, online adaptation, or difficulty-specific training.

## 8. Verification

Unit tests cover reset isolation, exact FIFO timing, update cadence, legal
actions, deterministic replay, config ordering, and a public-only checkpoint
boundary. Synthetic evaluation tests cover monotonic pass/fail cases, weak Easy
capability, incomplete schedules, per-opponent inversion, duplicate rows, and
hash tampering. A short real ViZDoom smoke precedes any formal validation run.
