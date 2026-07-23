# M5 Product-First Hybrid Governors

**Status:** Approved section by section by the project owner on 2026-07-23.

## 1. Context and decision

Bot Colosseo has one passing learned Aggressive vertical slice. Defensive and
Explorer each completed three auditable repair routes:

- offline distillation plus interpolation;
- closed-loop reward-shaped PPO V1;
- Teacher-assisted PPO V2 with a real 2,000-step CUDA preflight and a
  50,000-step bounded pilot.

V2 passed its engineering, supervision-coverage, checkpoint, and protocol
gates. Its frozen 20-episode validation smokes remained behaviorally negative:

| Style | Skill Retention | Primary metric | Point estimate | Decision |
|---|---:|---|---:|---|
| Defensive | 0.9231 | protective-presence delta | -0.0183 | `stop_50k` |
| Explorer | 0.7500 | route-entropy delta | -0.0631 | `stop_50k` |

Both estimates had the wrong sign, so the pre-approved rule prohibited a
continuation to 100,000 steps. The negative artifacts remain immutable.

The approved next route is product-first rather than another learned-policy
repair. Defensive and Explorer become transparent hybrid policies: the same
learned Strong Base is wrapped by a deterministic, fair-observation behavioral
governor. They are not described as pure reward-shaped RL policies.

## 2. Product objective and success definition

The public product must show four real policies in the same arena:

- Strong Base — learned policy;
- Aggressive — learned style policy;
- Defensive — hybrid behavioral governor;
- Explorer — hybrid behavioral governor.

Hybrid promotion is governed by product recognizability and capability
retention. Existing protective-presence, denial/recovery, route-entropy,
coverage, flank, and efficiency metrics remain frozen and are reported as
diagnostics. They are not hidden or silently weakened, but no single legacy
style metric is the only hybrid publication gate.

The hard hybrid gates are:

1. fair public observation and zero test-case access;
2. complete, protocol-clean paired evaluation;
3. Skill Retention of at least 0.85;
4. per-opponent retention of at least 0.75;
5. bounded, observable governor intervention;
6. stable visible differences in real recorded behavior;
7. anonymous user-study recognition at the thresholds in Section 8.

## 3. Architecture

Each hybrid policy contains one unchanged Strong Base and one small
deterministic governor:

```text
public observation -> frozen Strong Base CNN-GRU -> Base logits
        |                                           |
        +---- public temporal context --------------+
                                                    |
                                             Style Governor
                                                    |
                                      logit bias / bounded override
                                                    |
                                                final action
```

The governor can inspect:

- the current public frame;
- health, armor, ammo, own score, opponent score, and `has_core`;
- previous action;
- Base logits and recurrent features derived only from public observation;
- episode and decision counters;
- a bounded history of public scalar changes and executed actions;
- its own finite-state controller state.

It must not accept or derive behavior from:

- evaluation seed or case identity;
- learner side;
- map coordinates or region labels;
- privileged carrier identity;
- opponent health or hidden state;
- Teacher state;
- asymmetric Critic features.

Base logits and recurrent features are permitted because they are deterministic
functions of the same public Actor input. The governor does not access the
Critic.

## 4. Shared governor contract

The governor receives an immutable `PublicStyleContext` and returns an
auditable decision containing:

- the active state and public trigger;
- a finite per-action logit bias, or one legal override action;
- maximum remaining intervention decisions;
- the fallback condition;
- a compact reason code.

Bias is the default. Direct override is permitted only for short,
configuration-bounded behavior fragments. The policy falls back to the exact
Strong Base action when:

- the intervention reaches its consecutive limit;
- a stall detector fires;
- health crosses the safety threshold;
- carrying state conflicts with the active behavior;
- the governor has insufficient public evidence.

Every reset clears transient state while incrementing only the documented
episode counter. Identical public trajectories and reset order must produce
identical governor decisions.

## 5. Defensive governor

Defensive uses four states:

- `Base`;
- `Guard`;
- `Disengage`;
- `Recover`.

### 5.1 Score-triggered Guard

An increase in public `own_score` is reliable evidence that the Bot has just
completed an objective at its own base. The governor enters a short `Guard`
window at that point instead of attempting to infer location from privileged
coordinates.

Guard:

- suppresses sustained forward departure;
- applies alternating turn and short strafe biases to create visible scanning;
- preserves a high-confidence Base attack response;
- has a strict maximum duration;
- exits immediately if carrying begins, safety fallback fires, or the window
  expires.

### 5.2 Disengage and Recover

A sharp public health drop or low-health threshold enters `Disengage`.
The governor temporarily favors backward movement, strafing, and turning while
reducing forward-attack pressure. Stable health and the cooldown boundary move
the controller to `Recover`, which removes intervention and requires a short
Base-only interval before Guard can re-enter.

If `has_core` is true, Strong Base objective return has priority. Guard cannot
delay a carrying policy.

## 6. Explorer governor

Explorer leaves the pickup phase entirely to Strong Base. Route commitment
starts only on the public rising edge of `has_core`.

The initial route mode is:

```text
episode_counter mod 3
```

After a public own-score increase, the next commitment advances cyclically.
Modes are:

- Upper: light periodic bias toward `FORWARD_TURN_LEFT` and `STRAFE_LEFT`;
- Lower: the mirrored right-side pattern;
- Flank: a longer but bounded strafe, forward, and turn rhythm.

The Strong Base remains responsible for visual navigation and forward
progress. The governor shapes only the carrying trajectory. It does not infer
learner side or target coordinates.

Explorer states are:

- `Base`;
- `RouteCommit`;
- `StallRecovery`.

Commitment ends on score, drop, health safety fallback, stall, or its maximum
duration. `StallRecovery` runs Strong Base without route bias for a frozen
cooldown. This prevents repeated choreography from replacing task behavior.

## 7. Configuration and bounded selection

Defensive and Explorer have separate schema-validated YAML files. Configuration
contains only documented thresholds, bias magnitudes, rhythms, time limits,
cooldowns, and fallback limits.

Candidate selection uses a small validation-only grid frozen before real
evaluation. It may vary:

- two or three intervention magnitudes;
- two bounded window lengths;
- one or two safety thresholds.

It must not perform unconstrained search, access test cases, tune on user-study
answers, or select a candidate from a hand-picked video. Candidate identity
binds the Base checkpoint hash, governor config hash, code revision, scenario
hash, validation-case hash, and test-access flag.

## 8. Evaluation and promotion

### 8.1 Unit and interface gate

Tests must prove:

- every state transition and boundary;
- deterministic output for identical public trajectories;
- no seed, case, side, coordinate, region, or privileged parameter;
- finite bias and legal override actions;
- intervention and consecutive-override limits;
- exact Base fallback behavior;
- schema and checkpoint-hash rejection;
- complete telemetry reason codes.

### 8.2 Real 20-episode smoke

Each style runs the existing paired validation schedule:

- 20/20 complete episodes;
- zero protocol inconsistencies;
- Skill Retention at least 0.85;
- every opponent retention at least 0.75;
- nonzero but bounded intervention;
- no output overwrite and no test access.

Defensive must exercise score-triggered Guard and safety fallback. Explorer
must exercise all three modes and distinct carrying action signatures. Failure
stops the candidate.

### 8.3 Formal product evaluation

A passing smoke permits a 200-episode paired validation evaluation. It reports:

- task performance and per-opponent retention;
- governor intervention, override, fallback, and state occupancy;
- the unchanged legacy Defensive or Explorer metrics;
- action-signature differences from Strong Base;
- complete protocol and hash-chain evidence.

The selected policies then produce one MP4 each and a real four-policy 2-by-2
GIF through the existing showcase renderer.

### 8.4 Anonymous user study

At least ten anonymous participants view randomized, label-blinded clips.
The report includes sample size and uncertainty and does not claim statistical
significance from a small sample.

Promotion thresholds are:

- overall top-1 style recognition at least 60%;
- recognition for every individual style at least 50%.

Study answers cannot be used to retune the candidate evaluated by those same
answers.

## 9. Telemetry and evidence

Every hybrid action records:

- policy and episode identity;
- governor state;
- public trigger and reason code;
- Base action and final action;
- bias or override type;
- intervention duration;
- fallback reason;
- public `has_core`, score change, and health change.

Evidence lives under new paths:

- `runs/m5/hybrid/`;
- `reports/m5/hybrid/`;
- `docs/assets/showcase/`.

Old distillation, PPO V1, and PPO V2 artifacts are never overwritten. The
hybrid manifest binds telemetry, episode ledger, summary, media, configuration,
checkpoint, and code hashes.

## 10. Failure handling

The pipeline fails before evaluation if:

- a privileged field appears in a governor or public-policy interface;
- a config is incomplete, non-finite, or outside frozen bounds;
- a checkpoint, config, scenario, or case hash drifts;
- an action or bias is illegal or non-finite;
- immutable evidence already exists.

At runtime, expected behavioral uncertainty triggers deterministic Base
fallback and telemetry. Programming errors, invalid actions, or non-finite
values raise rather than silently falling back.

Environment retries and protocol inconsistencies retain their existing strict
accounting. A complete failed candidate is preserved.

## 11. Packaging boundary

Public documentation must distinguish:

- learned policies;
- hybrid governors;
- training success;
- product-recognition success;
- validation evidence;
- future test evidence.

The release package contains the Strong Base weights, governor configs, code
revision, and a hash-bound manifest. It must not present a governor config as a
new learned checkpoint or claim that failed V1/V2 style learning succeeded.

## 12. Completion boundary

The hybrid route is complete only when:

1. both 20-episode smokes pass;
2. both 200-episode product evaluations pass capability and protocol gates;
3. all legacy style diagnostics are reported honestly;
4. four real policy videos and the 2-by-2 GIF are hash-bound;
5. the anonymous study meets the recognition thresholds;
6. release-package verification and bilingual documentation pass.

Until then, Aggressive remains the only successful learned style vertical
slice, and M5/M6 remain incomplete.
