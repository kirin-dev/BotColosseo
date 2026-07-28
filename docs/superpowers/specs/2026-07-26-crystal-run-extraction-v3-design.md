# Crystal Run: Extraction Final Research Design

## Status

Approved design for the clean final research and showcase branch
`feat/crystal-run-extraction-v3`.

This branch is built from the completed Extraction v2 prototype, but its public
repository tells one story only. It does not present v1 or v2 milestones,
reports, media, failures, synthetic studies, or implementation chronology.
Those artifacts remain recoverable from their existing branches and Git
history.

## Research question

Can one strong fair-observation visual Bot be adapted into visibly
Aggressive, Defensive, and Explorer policies in a compact extraction game
while retaining useful task capability?

The method under test is:

```text
one frozen Strong Base checkpoint
  -> one learned residual style adapter per style
  -> shared task reward + bounded style reward + KL retention
  -> paired Style Fidelity x Skill Retention evaluation
```

The final release may claim that this method transfers to the extraction
scenario only if the predefined Strong, style-retention, reward-hacking, and
single-use official-test gates pass.

## Product boundary

The public product name is `Crystal Run: Extraction`. Version labels such as
v1, v2, and v3 remain Git-level development identifiers and do not appear in
the final README narrative, media titles, report stage names, or commands.

The public repository contains only:

- the extraction game and legal assets;
- one Strong Base and three learned style policies;
- training, validation, official-test, rendering, and audit code;
- final validation videos and official-test results;
- a concise bilingual README, method description, and reproduction commands.

It excludes milestone diaries, superseded reports, old media, synthetic user
studies, difficulty-control work, runtime style governors, and unrelated
experiments.

## Frozen game rules

The completed Extraction v2 mechanics are retained without behavioral
changes:

- synchronous fair-start 1v1 raids lasting 75 seconds;
- two neutral and side-symmetric extraction zones;
- both zones open at 30 seconds;
- extraction requires three uninterrupted seconds inside a zone;
- damage, leaving the zone, or attacking interrupts extraction;
- both players start with 100 HP and 30 rounds;
- every valid hit deals exactly 20 damage;
- no magazines, reload action, armor, healing, or respawn;
- two symmetric ammunition pickups add 10 rounds up to a 40-round cap;
- a three-slot backpack and item values 10, 25, and 50;
- deterministic replacement of the oldest minimum-value item by a better item;
- a dropped replacement remains available in the world;
- terminal death creates one corpse cache at the death position;
- corpse-cache items use the normal backpack replacement rules;
- kills and carried value do not score;
- only successfully banked extracted value determines the outcome.

Map geometry, dynamics, action space, observation semantics, item economy,
damage, timing, and win conditions are frozen. Engineering fixes are allowed
only when they preserve observable game behavior, such as eliminating stale
processes, nondeterministic case setup, protocol duplication, or logging
errors.

## Fair-observation boundary

The Actor and every style adapter may receive only:

- an `84x84` grayscale first-person frame;
- own HP and total ammunition;
- own carried total, free-slot count, and minimum slot value;
- extraction-open state and own extraction progress;
- own banked value, remaining time, and previous action.

They may not receive opponent HP, coordinates, angle, ammunition, backpack,
extraction progress, hidden item coordinates, region IDs, automap, depth,
labels, object buffers, viewer telemetry, or future state.

Privileged state is limited to scripted Teachers, the asymmetric Critic,
reward and event computation, offline metrics, deterministic integration
tests, and Oracle diagnostics. Typed Actor observation and privileged state
interfaces remain separate and are protected by leakage tests.

## Repository isolation

Before public cleanup, the completed v2 commit remains preserved by its
existing branch and Git history. The final branch deletes v1/v2 public
documentation, reports, media, configurations, and entry points that are not
required by the final extraction product.

The intended final structure is:

```text
README.md
README_CN.md
Plan.md
assets/scenarios/crystal_run_extraction/
configs/{train,validation,test,policies}/
src/botcolosseo/{envs,agents,training,evaluation,demo}/
scripts/
docs/assets/showcase/
reports/{validation,official-test}/
reports/release-manifest.json
```

Reusable implementation is retained only when it serves the final scenario.
Public names use `extraction` rather than historical milestone or version
names.

## Policy architecture

### Strong Base

Strong Base uses the existing recurrent visual Actor:

```text
public pixels -> visual CNN -> GRU -> Base policy logits
public scalars -----------^
```

It is the only shared capability source. Its checkpoint, configuration,
scenario, observation schema, and SHA-256 are frozen before style work starts.

### Learned style policy

Each style adds a small learned residual adapter:

```text
frozen Base logits + bounded residual delta logits -> style action
```

All styles must bind the exact same Strong Base checkpoint hash. Initial style
training freezes the visual CNN, GRU, and Base policy head and updates only the
residual adapter and its style policy head.

The GRU may be unfrozen once at a low learning rate only when validation shows
insufficient style separation while retention remains above its hard gate.
That choice must be logged and frozen before official test. The visual CNN
always remains frozen during style shaping.

Final public Bots must be learned policies. Runtime governors, scripted
Teachers, hand-coded state machines, privileged routing, and policy switching
are prohibited in released inference.

## Engineering-first, experiments-second sequence

All implementation, deterministic tests, resumable runners, evidence schemas,
and rendering code are completed before long experiments begin. Experiments
remain mandatory before the final release:

```text
engineering and short preflights
  -> Strong long training and validation
  -> Aggressive vertical slice
  -> Defensive and Explorer
  -> configured ablations and held-out validation
  -> freeze all artifacts
  -> one official test
  -> final videos, README, and release audit
```

The project is not considered complete merely because engineering or videos
exist. It is complete only after the required experiments and official test
have finished and the evidence-backed release has been audited.

## Strong Base training

Strong does not reuse v2 weights. It reuses only the scenario implementation,
model architecture, and training framework.

Training is:

```text
new privileged Strong Teacher demonstrations
  -> fair-observation behavior-cloning warm start
  -> recurrent PPO against a scripted opponent pool
  -> historical checkpoints plus lightweight PFSP
  -> frozen Strong Base
```

The common task objective prioritizes extracted value and successful
extraction. Dense progress shaping is bounded and decays during training.
Kills have no independent terminal reward. Wasteful combat, death, timeout,
and unbanked value loss are penalized.

Only positive loot/extraction progress bonuses decay. Death, unbanked-value
loss, timeout-value loss, and wasteful-combat penalties remain active for the
entire run; `shaping_decay` must never anneal terminal-risk penalties to zero.

The opponent pool covers RandomLegal, Aggressive, Defensive, Explorer,
generalist scripts, and historical Strong checkpoints. Side, spawn, and item
layout coverage are balanced.

Strong may receive one documented validation calibration. After that
calibration its configuration and checkpoint selection rule are frozen.

## Strong validation gate

Style work cannot start unless Strong satisfies all of:

- no-opponent extraction rate at least 90%;
- scripted-pool average win rate at least 70%;
- win rate against every major scripted opponent at least 55%;
- overall extraction rate at least 75%;
- held-out-layout extraction rate at least 70%;
- positive mean extracted-value advantage;
- zero protocol inconsistencies;
- zero Actor privilege violations;
- zero test-case access.

Reports include case-level rows, paired side swaps, opponent breakdowns, and
bootstrap confidence intervals. A reward curve, BC action accuracy, or one
successful video cannot substitute for this gate.

If the validation-ranked Strong candidate fails held-out or solo capability,
the selector proceeds in frozen rank order. It does not declare the whole run
failed while a lower-ranked candidate could still satisfy every Strong gate.

## Style objective

Every style optimizes:

```text
R = R_task
  + lambda_style * R_style
  - beta_KL * KL(style_policy || frozen_Strong_Base)
  - residual_magnitude_penalty
```

`R_task` is identical for Strong and every style. Style rewards are
opportunity-conditioned, bounded per event and per episode, and cannot create
terminal success independently of extraction.

Checkpoint selection uses the validation Pareto frontier between Style
Fidelity and Skill Retention. It never selects the checkpoint with the
largest raw style reward alone.

Validation-eligible Pareto candidates are checked against held-out capability
in frozen rank order. A held-out failure advances to the next eligible
candidate rather than incorrectly failing the entire style run.

### Aggressive

Aggressive rewards:

- early entry into meaningful contested areas;
- favorable encounter initiation given own HP and ammunition;
- bounded pursuit of a damaged opponent;
- valid hits rather than attack actions;
- kill-to-cache pickup and cache-to-extraction conversion.

It does not reward empty firing, impossible pursuit, attacks without an
opportunity, kills without loot conversion, or kills without extraction.

Aggressive is the required first vertical slice. Defensive and Explorer do
not start until Aggressive passes its complete validation gate.

### Defensive

Defensive rewards:

- risk reduction as carried value increases;
- disengagement at low HP or ammunition;
- value-preserving route and extraction choices;
- successful banking of meaningful carried value.

It does not reward empty-backpack camping, inactivity, avoiding all loot,
reduced attack count by itself, or timeout survival without extraction.

### Explorer

Explorer rewards:

- visiting distinct meaningful loot regions;
- taking alternate approaches;
- replacing low-value backpack items with better ones;
- additional search only when time and carried value justify it;
- stopping exploration and extracting under urgency.

It does not reward distance by itself, repeated region visits, turning in
place, low-value wandering, or continued search with a high-value backpack
near timeout.

## Style validation gate

Each style must satisfy all of:

- paired composite Skill Retention at least 85%;
- extraction rate no more than 10 percentage points below Strong;
- mean extracted value at least 85% of Strong;
- no catastrophic degradation against any major opponent;
- predefined style metrics move in the intended direction;
- the paired bootstrap interval supports the intended style direction;
- all reward-hacking counterexamples pass;
- zero protocol inconsistencies;
- zero Actor privilege violations;
- zero test-case access.

“Catastrophic degradation” is frozen as either more than a 20 percentage-point
win-rate loss from Strong against one scripted opponent or an absolute
opponent-specific win rate below 40%. Style direction uses a deterministic
10,000-resample paired bootstrap. Aggressive evidence requires an ordered
kill-to-cache-to-extraction conversion, Defensive evidence requires
opportunity-conditioned disengagement plus meaningful banking, and Explorer
evidence requires distinct loot regions and a genuine backpack-upgrade-to-
extraction conversion. Attack count and raw route distance are diagnostics,
never standalone style gates.

The initially frozen CNN/GRU/residual configuration is the primary method.
Configured reward-only and reward-plus-KL runs are later ablations of the same
training system, not additional product architectures. Full-policy PPO
fine-tuning and style-conditioned policies are out of scope.

## Splits and paired evaluation

Evaluation roles are separated before long training into:

- `train`;
- `validation`;
- held-out-configuration validation;
- a sealed `official test` created only after all four policies are selected.

Strong and all styles use the same paired seed, learner side, opponent,
layout, spawn, episode budget, and scenario hash within each evaluation case.
Side-swapped pairs and opponent types are balanced.

The intended evaluation sizes are:

- 240 validation episodes per policy;
- 40 additional idle-opponent validation episodes for Strong capability;
- 120 held-out-configuration episodes per policy;
- 400 official-test episodes per policy;
- 1,600 official-test episodes across Strong and three styles.

Headline common metrics are win rate, extraction rate, mean extracted value,
survival rate, timeout value loss, and worst-opponent performance. Style
reports add opportunity-normalized encounter, disengagement, route,
backpack-upgrade, and conversion metrics.

## Official-test discipline

Before test:

1. freeze the scenario WAD and observation schema;
2. freeze Strong and all three style checkpoints;
3. freeze configs, thresholds, metrics, and selection rules;
4. generate a new side-balanced official-test case manifest from system
   entropy; test cases do not exist in train or validation configuration;
5. bind the sealed case-manifest hash and all other hashes into a
   release-candidate manifest;
6. audit that no official-test episode has been executed.

Official test is then run once. After it starts, only aggregation, plots,
documentation, and release packaging may change. Policies, rewards,
thresholds, and case definitions may not change.

If a test gate fails, the result is reported as a failed research gate. The
same test cannot become another tuning split. A subsequent method revision
requires a new versioned protocol and a new untouched test set.

## Showcase

Showcase videos are automatically selected from the frozen validation
manifest, never from official test. They are real policy replays, not scripted
animations.

The final public set contains:

- one Strong generalist replay;
- one Aggressive five-hit kill, corpse-cache pickup, and extraction chain;
- one Defensive meaningful-value protection and safe extraction;
- one Explorer alternate route, genuine backpack upgrade, and extraction;
- one concise comparison image or video;
- one official-test metric figure;
- one method figure.

Viewer overlays may show both HP bars, hit damage, ammunition, backpack slots,
cache transfer, extraction progress, and final banked value. Those fields
remain absent from Actor inputs.

The README is concise and bilingual. Its order is:

1. product and game loop;
2. four real Bots and videos;
3. one-Base-to-three-styles method;
4. official-test results;
5. fair-observation boundary;
6. reproduction commands;
7. honest limitations.

It contains no v1/v2 history, milestone chronology, synthetic perception
study, difficulty matrix, or unsupported claim.

## Reliability and artifact handling

Every long runner is resumable and records:

- immutable config and scenario hashes;
- PID, device, start time, and exit code;
- stage-specific logs;
- checkpoint hashes and update counts;
- split and case-manifest hashes;
- test-access state.

Runners refuse to overwrite completed artifacts and reject ambiguous partial
directories. A short preflight must pass before every long run. A process is
treated as healthy only after a delayed PID, log, GPU, and error check.

Tracked reports remain compact. Large checkpoints, replay ledgers, and raw
training data stay outside normal Git history and are bound through SHA-256
release records.

## Explicit exclusions

The final version does not add:

- new game mechanics or maps;
- full-policy style fine-tuning;
- a unified style-conditioned policy;
- runtime governors;
- difficulty levels;
- a Web application;
- human or synthetic user studies;
- LLMs, VLMs, world models, or diffusion policies;
- teams, squads, persistence, shops, armor, healing, reloads, or inventory UI.

## Completion definition

Crystal Run: Extraction is complete only when:

1. the clean repository contains no public v1/v2 story or obsolete artifact;
2. frozen mechanics and fair-observation gates pass;
3. Strong passes its full capability gate;
4. Aggressive, Defensive, and Explorer derive from the same Strong hash;
5. all three pass style, retention, and reward-hacking validation gates;
6. configured ablations and held-out validation finish;
7. the single-use official test finishes without integrity violations;
8. final videos and figures are generated from frozen evidence;
9. the bilingual README contains only supported claims;
10. an independent release audit rehashes and validates every published
    artifact.
