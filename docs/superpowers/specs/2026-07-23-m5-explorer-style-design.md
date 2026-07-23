# M5 Explorer Style Design

**Status:** approved by the project owner
**Date:** 2026-07-23  
**Scope:** one Explorer checkpoint, frozen validation evidence, and showcase assets

## 1. Product outcome

Explorer must visibly choose more effective routes while retaining the same
core-delivery objective as Strong Base. It is not a wandering bot. The intended
short demonstration shows one checkpoint completing successive deliveries by
the upper, lower, and flank routes with only a bounded efficiency cost.

The first route is a score-conditioned route Teacher, success-filtered
distillation, and fixed adapter-strength interpolation. PPO is excluded from
this first attempt and may be proposed only if the frozen evaluation shows that
the learned route style is too weak.

## 2. Legal runtime behavior

The route for a delivery is selected from:

```text
(own_score - episode_initial_own_score) % 3 == 0 -> direct_upper
(own_score - episode_initial_own_score) % 3 == 1 -> direct_lower
(own_score - episode_initial_own_score) % 3 == 2 -> flank
```

`own_score` and `has_core` are already legal Actor scalars. The initial own
score is observed at the recurrent episode boundary and retained in public
history. The visual frame, previous action, and recurrent state provide local
navigation context. The policy receives no route ID, random seed, coordinate,
region, automap, or privileged state at inference.

The Teacher may use privileged geometry to steer through route waypoints.
Opponent-side geometry is an exact mirror of host-side geometry. A delivery
uses the selected route while approaching the core and the reversed route
while carrying it home. A new route is selected only after the public own score
changes.

This deterministic public-state rule avoids contradictory labels from hidden
random route choices and remains reproducible under greedy checkpoint
evaluation.

## 3. Considered approaches

### A. Score-conditioned route distillation — selected

Use the public score as a stable route-cycle key, retain only successful
deliveries, replay Strong Base behavior outside the navigation window, and
interpolate the resulting style branch. This is the shortest route to a
learned, legible, independent checkpoint.

### B. Runtime action sampling

Increasing policy temperature could produce different paths quickly, but the
variation would come from inference randomness rather than the Explorer
checkpoint. It would also make demos and metrics less reproducible.

### C. Coverage-reward PPO

Reward-shaped PPO matches the long-term formulation but is slower and can
reward loops, excessive turning, or inefficient detours. It remains a bounded
fallback after the first route has complete evidence.

## 4. Training-only route Teacher

The Teacher holds a side-normalized ordered waypoint list for the route
selected by the public own-score increase from the episode boundary.

- Without the core, it advances from home toward the core through the selected
  route.
- With the core, it traverses the route in reverse toward home.
- A waypoint advances only after the learner enters a fixed local radius.
- Death or a score resets route progress from the new public state.
- The Teacher never receives positive credit for turning, region entry, time
  alive, or distance traveled by itself.

The Teacher labels navigation only. Strong Base supplies context actions for
combat, low-health recovery, and steps outside a retained successful delivery.
This keeps route style separate from general capability.

## 5. Success-filtered dataset

Generate exactly 50,000 train-only transitions across all five frozen opponent
types and both learner sides. A candidate delivery window begins after a score
or episode reset and ends on learner score, opponent score, death, or episode
end.

Positive navigation labels are retained only from windows that end with a
learner SCORE and whose visited region signature matches the selected route:

- `upper_route` for `direct_upper`;
- `lower_route` for `direct_lower`;
- both `flank_west` and `flank_east` for `flank`.

Failed, mixed, stalled, or unscored paths are logged but receive no Explorer
positive labels. To prevent the easiest route from dominating the
success-filtered set, retain at most the first 15 verified successful windows
per route in the frozen case order. Record both raw and retained route counts.
Strong Base replay supplies 25%–50% of supervised context.

The production data gate requires:

- exactly 50,000 public-observation transitions;
- all five opponents and both learner sides;
- successful examples from all three routes;
- each route at least 20% of successful delivery windows;
- at least 1,000 selected route-navigation transitions;
- context replay between 25% and 50%;
- zero privileged Actor fields and `test_cases_accessed: false`;
- hashes for the scenario, train cases, Base checkpoint, configuration, and
  every shard.

A failed gate preserves its evidence and prevents training.

## 6. Distillation and fixed interpolation

Initialize an Explorer residual style adapter and copied policy head from the
same Strong Base checkpoint. Freeze the complete Base Actor and train only the
adapter and Explorer policy head for 1,000 updates.

The loss is masked cross-entropy on successful route labels plus the existing
KL constraint on Strong Base replay. The offline gate requires:

- finite loss and gradients;
- route-target agreement at least 0.20 above the neutral branch;
- Base-replay argmax action drift no greater than 0.10;
- byte-identical frozen Base Actor tensors;
- matching data, scenario, configuration, and checkpoint hashes.

Create immutable neutral-to-distilled interpolation checkpoints at:

```text
alpha = 0.25, 0.50, 0.75
```

Run a protocol-clean 20-episode validation smoke for each. Only candidates
passing every hard gate are eligible. Rank eligible candidates by higher
normalized Route Entropy shift, then higher Skill Retention, then lower alpha.
The grid and ranking rule are frozen before results are observed.

## 7. Route attribution and frozen evaluation

Evaluation uses privileged region IDs only for offline scoring. They never
enter the Actor. For every learner SCORE, the evaluator attributes the
preceding carried-core path:

1. `flank` if both flank regions were visited;
2. `upper` if `upper_route` was visited without a flank signature;
3. `lower` if `lower_route` was visited without a flank signature;
4. `mixed_or_unknown` otherwise.

Only completed upper, lower, or flank deliveries contribute positive route
credit. Mixed, unscored, looping, and incomplete paths remain visible in the
ledger.

For the three credited routes, normalized entropy is:

```text
H(route) / log(3)
```

The primary style statistic is the paired Explorer-minus-Strong-Base
difference in per-episode normalized route entropy. A fixed-seed paired
bootstrap produces its 95% confidence interval.

The formal evaluation uses ten side-swapped validation pairs against each of
the five opponents: 200 episodes total. It passes only when:

1. all 200 unique rows are complete and protocol-clean;
2. overall Skill Retention is at least 0.85;
3. per-opponent retention is at least 0.75;
4. the Route Entropy shift confidence-interval lower bound is above zero;
5. Explorer completes all three credited route types;
6. Explorer flank completion rate is greater than Strong Base;
7. objective completion is at least 0.85 times Strong Base;
8. decisions per learner score are at most 1.35 times Strong Base;
9. scenario/checkpoint hashes match and no test case is accessed.

Zero-score episodes receive zero route entropy and remain subject to the skill,
objective, and efficiency gates. This prevents failed wandering from being
silently dropped.

## 8. Anti-hacking rules

- Region entry without a subsequent learner SCORE earns no route credit.
- Raw turning, distance, survival time, and repeated region entry earn nothing.
- Repeated loops do not increase the set-based route signature.
- A mixed path is not relabeled as the closest successful route.
- A long flank path must still satisfy objective and decisions-per-score gates.
- Validation metrics cannot select new alpha values or alter route thresholds.
- Runtime action overrides, stochastic temperature changes, privileged
  inference, and hand-picked evaluation cases are forbidden.

## 9. Components and artifacts

Implementation reuses the existing region graph, synchronous duel environment,
demonstration schema, residual style model, distillation trainer,
interpolation machinery, paired evaluation pattern, and evidence audit.

New components are limited to:

- the score-conditioned Explorer route Teacher;
- successful route-window labelling and dataset generation;
- Explorer-neutral interpolation identity;
- route trace attribution, paired metrics, selection, and evidence audit;
- one production pipeline and configuration;
- checkpoint-bound Explorer video, route heatmap, and metric card.

The final showcase combines Strong Base, Aggressive, Defensive, and Explorer
only when each published claim is backed by its own evidence. A provisional
checkpoint may be shown only with an explicit failed-gate label.

## 10. Error handling and resumability

- Dataset shards, ledgers, summaries, and manifests are hash-bound.
- Existing output is resumed only when config, scenario, Base, cases,
  checkpoint, and estimator identities match exactly.
- Duplicate episode keys, malformed region traces, partial side swaps,
  non-finite training values, frozen-Base drift, or test access are fatal.
- ViZDoom warm-up failures may retry at most twice and are counted.
- A complete failed gate exits nonzero without deleting or rewriting the
  threshold.

## 11. Verification and execution

Before production compute, unit and real-environment smokes must prove:

- score-to-route cycling, route reversal, and side mirroring;
- waypoint progress and reset behavior;
- successful, failed, mixed, and looping window labels;
- exact route-balance and context-replay data gates;
- only adapter and Explorer policy-head parameters change;
- interpolation hashes and frozen Base tensors are correct;
- route attribution, entropy, retention, efficiency, pairing, retry, resume,
  split, and evidence-integrity checks fail closed;
- a small CUDA/ViZDoom run produces finite updates and a loadable checkpoint.

The production command, log, PID, progress, artifact, and verification commands
will be recorded in `script.md`. Under the current project rule, Codex launches
required detached experiments, checks them once or twice, and then leaves them
running without blocking unrelated development.
