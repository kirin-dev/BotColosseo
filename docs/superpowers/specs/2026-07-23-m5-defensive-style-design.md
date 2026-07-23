# M5 Defensive Style Design

**Status:** approved direction; written specification awaiting final review  
**Date:** 2026-07-23  
**Scope:** one Defensive checkpoint, its frozen validation gate, and evidence

## 1. Product outcome

Defensive is the second player-recognizable Bot style derived from the same
fair-observation capability anchor used by Aggressive. Its primary visible
behavior is protecting the home approach, denying an opponent carrying the
core, and recovering a dropped core. Low-health disengagement is secondary.

The desired short demonstration is: the opponent carries the core toward the
learner's base, Defensive intercepts, forces a drop, and recovers possession.
It must remain capable of proactively scoring. Standing at home without a
current defensive risk is explicitly a failure, not a style success.

This design covers Defensive only. Explorer, difficulty control, the four-way
M5 showcase, and user study remain separate dependent work.

## 2. Constraints and non-goals

- The final Actor consumes only the existing public visual observation,
  scalar status, and recurrent history. Coordinates, regions, carrier identity,
  opponent health, and event look-ahead remain privileged training/evaluation
  signals.
- The Strong Base CNN and recurrent trunk remain frozen. Defensive trains a
  residual adapter and copied policy head in an independent checkpoint.
- The frozen `defensive_script` opponent is not modified. A new
  training-only protective Teacher prevents benchmark and historical-opponent
  drift.
- M2/M3 test data and the final test split are not accessed. All selection is
  train plus validation only.
- No long PPO phase is part of the first route. It may be proposed only after
  the approved distillation/interpolation route produces complete evidence.
- Runtime action overrides, privileged inference, manual highlight selection,
  and reward for raw turning or idling are forbidden.

## 3. Considered approaches

### A. Risk-conditioned protection and recovery — selected

Train from successful protection windows, mix no-risk Strong Base behavior,
and select a fixed adapter-strength interpolation. This is event-grounded,
visually legible, and directly guards against camping.

### B. Low-health survival

Reward early disengagement and longer survival. This is tactically sensible
but difficult for a viewer to recognize as Defensive in a short clip, and it
depends on relatively sparse health trajectories.

### C. Broad weighted Defensive reward

Optimize zone occupancy, recovery, denial, survival, and disengagement in one
PPO objective. It covers more of the long-term plan, but its signals compete,
its result is harder to diagnose, and Aggressive already showed that a long
PPO phase can erase deterministic style behavior.

Approach A is selected. It tests one clear product hypothesis before adding
more training complexity.

## 4. Side-normalized defensive state

All privileged geometry is mirrored by learner side before use. Let the
learner's home point be `(-640, 0)` for host and `(640, 0)` for opponent.

A decision is **at defensive risk** when at least one condition holds:

1. the opponent carries the core;
2. the uncarried core has `x < -128` for host or `x > 128` for opponent;
3. the opposing player has `x < -128` for host or `x > 128` for opponent while
   the learner is not carrying.

The **protective zone** is the area within 512 map units of the learner's home
point. The radius includes the home room and its immediate approaches rather
than rewarding a single camping coordinate.

A decision is an **unnecessary guard** only when the learner is inside the
protective zone, is not carrying, and none of the three risk predicates holds.
This definition makes the anti-camping gate an explicit counterfactual rather
than simply limiting total time near home.

## 5. Training-only protective Teacher

The new Teacher has three ordered modes:

1. **Recover:** at health at or below 25 while disadvantaged, disengage toward
   home until the immediate threat is broken.
2. **Protect:** when the opponent carries, intercept the carrier and use the
   existing legal combat macros; when the core is loose in the defensive half,
   move to recover it.
3. **Objective:** when there is no risk, follow the existing objective-first
   behavior and attempt to score.

The Teacher may read privileged state, but only its action paired with the
learner's public observation enters the dataset. Its name and artifacts are
separate from the frozen `defensive_script` opponent.

## 6. Success-filtered dataset

Generate 50,000 train transitions across the frozen five-opponent distribution
and both learner sides. A protection window begins when defensive risk becomes
true and ends on one of: opponent score, opponent drop, learner recovery,
risk resolution, death, or episode end.

Retain protective Teacher actions only from windows with at least one verified
outcome:

- a learner hit forces the opponent carrier to drop;
- the learner picks up a core dropped or left in its defensive half;
- the risk resolves without an opponent score and the learner resumes
  objective progress within 24 decisions.

No-risk supervision comes from the Strong Base action, not from a home-holding
Teacher. It must form 25–50% of selected transitions. The production data gate
requires:

- exactly 50,000 public-observation transitions;
- at least 1,000 selected risk-conditioned transitions;
- both sides and all five opponents represented;
- at least 100 completed denial or recovery windows;
- rejected-window and no-risk counts logged;
- scenario, case-manifest, base-checkpoint, and shard hashes bound;
- `test_cases_accessed: false`.

Failure preserves the dataset report but prevents training.

## 7. Adapter distillation and fixed strength grid

Initialize the existing residual style adapter and copied policy head from the
Strong Base; keep the base Actor byte-identical. Train for 1,000 updates using
cross-entropy on the success-filtered actions plus the existing
`D_KL(style || base)` constraint on no-risk replay.

The offline gate requires finite loss, hash-matched inputs, at least +0.20
absolute agreement with successful protective actions relative to the neutral
branch, and a no-risk home-directed-or-idle prediction rate no more than 0.05
above the Strong Base rate.

Create three immutable checkpoints by interpolating only `adapter.*` and
`policy.*` between the neutral initialized style branch and the distilled
branch:

```text
alpha = 0.25, 0.50, 0.75
```

The Strong Base actor tensors must remain byte-identical. The grid is fixed
before validation and cannot be extended after results are observed. A
protocol-clean 20-episode smoke is run for each point. Candidates are ranked:

1. all hard gates pass;
2. higher risk-conditioned protective-presence shift;
3. if shifts differ by at most 0.05 absolute rate, higher Skill Retention;
4. lower alpha as the deterministic tie-break.

Only an all-gate smoke candidate may enter the formal 200-episode evaluation.
If none passes, preserve the results and request approval before introducing
PPO, changing thresholds, or testing more alpha values.

## 8. Frozen evaluation metrics

Evaluation pairs Strong Base and Defensive on ten side-swapped validation pairs
against each of the five frozen opponents: 200 episodes total. The append-only
ledger includes task results plus these privileged offline counters:

- risk decisions;
- protective-zone decisions under risk;
- opponent-carrier opportunities;
- carrier denials caused by the learner;
- defensive-half recovery opportunities and learner recoveries;
- unnecessary-guard decisions;
- low-health disengagement opportunities and successful escapes.

The primary style statistic is the paired difference in risk-conditioned
Protective Presence Rate. Denial and recovery are secondary event-grounded
checks. The formal gate passes only when all conditions hold:

1. exactly 200 unique rows, correct side swaps, no protocol inconsistency;
2. overall Skill Retention at least 0.85;
3. per-opponent retention at least 0.75;
4. the 95% paired-bootstrap interval for Protective Presence shift is above 0;
5. combined denial/recovery rate is greater than Strong Base in the observed
   schedule;
6. unnecessary-guard rate is at most 0.20;
7. Defensive objective completion is at least 0.85 times Strong Base objective
   completion;
8. zero test-case access and matching scenario/checkpoint hashes.

Rare-event denominators are reported as absolute counts and rates; a zero
denominator cannot silently pass the denial/recovery condition.

## 9. Reward-hacking and failure rules

Every training reward or selection counter has a precondition, per-episode cap,
logged component, and counterexample test. In particular:

- home occupancy earns no benefit without defensive risk;
- idling never earns a positive reward;
- attacking without a carrier or defensive threat does not count as denial;
- a recovery counts only after a verified loose-core opportunity;
- prolonging a threat cannot accumulate uncapped protective reward;
- sacrificing scoring for indefinite defense is caught by retention and
  objective-completion gates.

A checkpoint that looks defensive in one video but fails a frozen metric is
retained as diagnostic evidence and is not published as Defensive.

## 10. Components and artifacts

Implementation reuses the existing style model, checkpoint loader,
interpolation machinery, synchronous duel runtime, and append-only evidence
patterns. New work is limited to:

- side-normalized defensive predicates and counters;
- the training-only protective Teacher;
- success-window dataset generation and distillation configuration;
- neutral-to-distilled fixed interpolation support;
- Defensive paired evaluation and audit;
- `configs/m5/defensive*.yaml`, `reports/m5/defensive/`, and a milestone record.

No generalized style framework or unrelated refactor is required. Shared code
is extended only where Aggressive assumptions currently prevent Defensive use.

## 11. Error handling and resumability

- Dataset shards, training checkpoints, smokes, and formal rows are append-only
  or atomically replaced with hash-bound manifests.
- A rerun resumes only when config, scenario, base, cases, and source hashes
  match exactly.
- ViZDoom warm-up failures may retry at most twice and must be counted.
- Duplicate episode identities, partial side swaps, NaN losses, checkpoint
  drift, test access, or privileged Actor input are fatal.
- A complete but failed experimental gate exits nonzero without deleting
  evidence or changing thresholds.

## 12. Verification and handoff

Before production compute, tests must prove:

- host/opponent risk predicates are mirrored correctly;
- no-risk home occupancy is classified as unnecessary guarding;
- the Teacher pursues the objective when safe and protects only under risk;
- success-window inclusion/exclusion and event attribution are correct;
- reward caps and all reward-hacking counterexamples hold;
- the frozen base actor is byte-identical after distillation/interpolation;
- evaluator completeness, paired bootstrap, retry, resume, hash, and split
  checks reject malformed evidence;
- a real 1,024-step CUDA/ViZDoom smoke produces finite updates and counters.

The exact long-run launch, PID, log, progress, resume, artifact, and success
commands are recorded in `script.md`. Under the current project rule, Codex
launches any required `nohup` experiment, checks it one or two times, and then
blocks the goal while the detached process continues.

Defensive is complete only after the frozen 200-episode gate passes and a real
Strong Base/Aggressive/Defensive comparison can be generated from checkpoint-
and-log-bound validation evidence.
