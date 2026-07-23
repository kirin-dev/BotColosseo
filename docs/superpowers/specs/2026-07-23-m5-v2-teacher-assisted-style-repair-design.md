# M5 V2 Teacher-Assisted Style Repair

**Status:** Approved section by section by the project owner on 2026-07-23.

## 1. Product objective

Bot Colosseo must still demonstrate one fair-observation Strong Base shaped
into player-recognizable Aggressive, Defensive, and Explorer policies.
Aggressive already provides the complete vertical slice. M5 V2 repairs
Defensive and Explorer without changing the scenario, Strong Base, validation
cases, frozen evaluators, acceptance thresholds, or public Actor observation.

This is a product and mechanism repair, not a claim of a new reinforcement
learning algorithm. The implementation combines a frozen learned capability
backbone, lightweight style branches, on-policy PPO, and training-only Teacher
regularization.

## 2. Evidence for changing the route

The first closed-loop repair trained each style for 200,000 real environment
steps. Both runs completed without KL early stops, retained task skill, and
produced complete protocol-clean 20-episode validation smokes. Both moved the
primary style metric in the wrong direction.

### 2.1 Defensive

- Skill Retention was `0.8974`.
- Protective-presence delta was `-0.0875`, with 95% interval
  `[-0.3345, 0.0295]`.
- `carrier_denial` and `defensive_recovery` never appeared in the production
  training reward-component ledger.
- Scaled positive presence, entry, and resolution rewards accumulated to about
  `+68.6`; unnecessary-guard and concession penalties accumulated to about
  `-152.7`.

The easiest policy response was therefore to avoid defensive risk, not to
resolve it. More steps with the same objective would reinforce the same
failure.

### 2.2 Explorer

- Skill Retention was `0.9500`.
- Route-entropy delta was `-0.1262`, with 95% interval `[-0.3155, 0]`.
- The styled policy completed no flank route and had lower mean unique-region
  coverage than Strong Base.
- The training target always began with Upper, moved to Lower after one score,
  and exposed Flank only after two scores.
- Production training averaged about 1.9 learner scores per episode.

The reward ledger proved that some target routes scored, but did not prove
that Flank was reachable often enough. A hidden target that is usually
unreachable cannot reliably create diverse closed-loop behavior.

## 3. Alternatives

### 3.1 Selected: Teacher-assisted PPO with an internal Explorer route mode

Keep the fair-observation Strong Base frozen. Train small style branches with
PPO task returns, Base KL retention, and masked on-policy Teacher labels.
Explorer receives three route branches selected by a fair internal mode.

This directly addresses both observed mechanism gaps while preserving the
project's learned-policy and product-control story.

### 3.2 Rejected: reward and curriculum changes only

This is smaller, but Defensive's causal events are too sparse and Explorer's
route intent is not identifiable from the current objective. A second
reward-only run has a high probability of consuming another long budget
without producing a visible style.

### 3.3 Rejected: scripted wrappers around Strong Base

Scripted high-level wrappers could create fast visual differences, but would
weaken the central claim that styles are learned derivatives of the same
capability policy. They remain diagnostic Teachers, not publication Bots.

## 4. Shared V2 architecture

The visual CNN and recurrent GRU from the Strong Base remain frozen. V2 adds
only style-specific residual adapters and policy heads.

```text
public frame/scalars/previous action
                 |
       frozen visual CNN + GRU
                 |
       +---------+-------------------+
       |                             |
  Base policy head              style branch
                                     |
                     +---------------+---------------+
                     |                               |
             Defensive head               Explorer route heads
                                            Upper / Lower / Flank
```

The asymmetric Critic may continue to use privileged training state. No
privileged value enters the Actor, style branch, route-mode controller, or
published checkpoint inference input.

### 4.1 Defensive branch

Defensive retains one residual adapter and one policy head. During rollout, a
training-only `ProtectiveDefensiveTeacher` labels the current learner state.
Cross-entropy is applied only when the frozen `defensive_risk` predicate is
true. Base KL remains active across public observations and is the primary
regularizer outside risk states.

### 4.2 Explorer branches

Explorer contains three lightweight residual adapter/policy-head branches
sharing the same frozen CNN-GRU:

- Upper;
- Lower;
- Flank.

The active branch is a Bot-internal product state:

```text
initial_mode = persistent_episode_counter mod 3
active_mode = (initial_mode + own_score_progress) mod 3
```

The persistent counter increments once per reset. `own_score_progress` is
derived from the public own score relative to its value at reset. The policy
does not use the evaluation seed, case ID, map coordinates, region ID, or
hidden opponent state to choose a route. The rule is deterministic,
case-independent, and recorded in the checkpoint manifest.

A score changes the active branch, allowing multiple credited routes within
one episode. This directly supports the frozen per-episode route-entropy
estimator rather than merely distributing one fixed route across episodes.

## 5. Training data flow

Each on-policy token has two distinct paths:

```text
public observation -> style policy -> executed action -> PPO task return
privileged state   -> training Teacher -> auxiliary action label
```

Teacher actions are labels only. They are not executed and are never copied
into Actor observations. The recurrent rollout stores:

- the executed policy action and old log probability;
- the Teacher target action;
- a style-specific supervision mask;
- the active Explorer route mode, when applicable;
- existing public and privileged tensors used by PPO and the asymmetric
  Critic.

The fixed objective is:

```text
L = L_PPO + beta_kl * KL(style || base) + eta_aux * CE_teacher
```

The auxiliary loss is normalized over supervised tokens only. An empty mask
is a failed preflight, not a zero-valued successful update.

### 5.1 Defensive training distribution

Training remains train-split-only and side-swapped. Style V2 increases the
frequency of `objective_first` and `aggressive_script`, which create meaningful
defensive risk, while retaining other scripts and historical checkpoints for
capability coverage.

The reward keeps opportunity-conditioned positive presence, entry, denial,
recovery, and resolution terms. Unnecessary-guard and concession penalties are
reduced so their observed aggregate magnitude cannot dominate all positive
defensive signal. Task reward and the frozen evaluator still penalize
concessions and idle guarding.

### 5.2 Explorer training distribution

A route-specific `RouteExplorerTeacher` labels the branch selected by the
internal mode. Every short preflight must expose all three modes. The reward
retains target-region and target-route completion terms and adds a capped bonus
for the first completion of each distinct route within an episode.

Evidence is recorded separately for Upper, Lower, and Flank:

- supervised tokens;
- Teacher agreement;
- target-region rewards;
- target-route-score rewards;
- completed routes.

Aggregate reward is insufficient evidence when any mode is empty.

## 6. Gate-driven compute budget

Defensive and Explorer may run in parallel on the two available A100 GPUs.
Each style follows the same bounded schedule.

### 6.1 Real 2,000-step preflight

The run must prove:

- finite PPO, KL, auxiliary, value, entropy, and gradient metrics;
- zero test-case access;
- nonempty Defensive risk supervision;
- nonempty supervision and reward evidence for all three Explorer modes;
- a loadable hash-bound V2 checkpoint;
- no privileged field accepted by the public Actor interface.

Failure stops before a production pilot.

### 6.2 50,000-step pilot

The first immutable candidate is evaluated on the existing 20-episode frozen
validation smoke.

- If every gate passes, select the 50k checkpoint and do not spend the
  remaining budget.
- If Skill Retention passes and the primary style point estimate has the
  correct sign, but statistical or coverage gates remain inconclusive, one
  continuation to 100k is allowed.
- If the primary style estimate has the wrong sign, required mode/event
  coverage is absent, protocol integrity fails, or capability retention fails,
  stop without continuation.

### 6.3 100,000-step hard ceiling

The resumed candidate receives one final 20-episode smoke. There is no third
candidate and no automatic extension beyond 100k. A failure at this ceiling is
retained as the final V2 result and requires a new owner-approved design before
additional training.

### 6.4 Formal evaluation

A passing smoke permits the unchanged 200-episode validation evaluator. Its
complete result is immutable. A passing formal style gate permits the
corresponding 600-episode native Style x Difficulty block. A failed formal
result is not permission to alter thresholds, estimators, cases, or evidence.

## 7. Checkpoint and resume identity

V2 uses new run, report, and checkpoint directories and never overwrites V1.
Every checkpoint and manifest binds:

- V2 architecture and branch order;
- Strong Base checkpoint hash;
- warm-start hash;
- scenario hash;
- train-case manifest hash;
- opponent schedule hash;
- Teacher implementation/config hash;
- reward config hash;
- auxiliary-loss coefficients;
- Explorer route-mode rule;
- environment steps and continuation parent hash;
- `test_cases_accessed: false`.

Resume refuses any identity drift. The published Explorer checkpoint must
contain all three route branches and load through a public-observation-only
policy class.

## 8. Failure handling

The pipeline aborts before a long run when:

- any Actor path accepts privileged state;
- a supervision mask is empty;
- an Explorer mode lacks preflight evidence;
- a required checkpoint branch or hash is missing;
- any loss, gradient, KL, or reward value is non-finite;
- resume identity changes;
- an output directory already contains immutable evidence;
- a validation evaluator accesses test cases.

Environment retry behavior and protocol inconsistency accounting remain
unchanged. Complete failed artifacts are preserved.

## 9. Verification

### 9.1 Unit tests

- Explorer mode starts in round-robin order and advances on public score only.
- Case ID, evaluation seed, privileged position, and region do not affect
  branch selection.
- All three branches share the frozen Base and have independent trainable
  adapter/head parameters.
- Masked auxiliary loss rejects empty masks and ignores unsupervised tokens.
- Defensive labels appear only in frozen risk states.
- V2 checkpoint load/resume rejects missing branches and identity drift.
- Public Actor forward signatures and tensors remain privilege-free.

### 9.2 Real integration checks

- 2k CUDA/ViZDoom preflights satisfy the evidence in Section 6.1.
- Both physical GPUs can run one isolated style without path or device
  collision.
- Each selected checkpoint completes a real 20-episode validation smoke with
  zero protocol inconsistency before formal evaluation.

### 9.3 Frozen product gates

The existing Defensive and Explorer evaluators, retention thresholds,
anti-hacking checks, bootstrap estimators, validation cases, and decision
limits remain unchanged.

## 10. Completion boundary

M5 and M6 become complete only when:

1. Defensive and Explorer each pass their 200-episode formal style gate.
2. Aggressive, Defensive, and Explorer difficulty blocks produce 1,800
   complete episodes and pass the combined audit.
3. A real four-policy 2-by-2 GIF, one MP4 per policy, metrics card, and
   hash-bound publication manifest are generated.
4. The small anonymous user study is completed and reported within its stated
   evidence limits.
5. The four inference checkpoints pass release-package verification.
6. Bilingual documentation and one-command evaluation/demo paths bind the
   final hashes.

If V2 fails at the 100k ceiling, the repository retains Aggressive as the
successful vertical slice and presents both repair attempts as negative
mechanism evidence. It does not relabel incomplete styles as successful Bots.
