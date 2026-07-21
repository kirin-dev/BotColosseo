# Milestone 3 Design: Auditable League Training and Strong Base

**Status:** Approved architecture elaboration. This design implements the M3
section of `Plan.md` without changing the product story, the 13-action contract,
the legal Actor observation boundary, or the M2 frozen evaluator.

## 1. Outcome and non-goals

M3 turns the provisional M2 PPO policy into an auditable Strong Base by adding:

1. a versioned historical policy pool with immutable checkpoint provenance;
2. a deterministic mixture of script opponents, PFSP-selected historical
   opponents, and uniform historical coverage;
3. validation-only pool admission and a complete historical cross-play payoff
   matrix;
4. held-out, worst-case, and no-opponent capability evaluation;
5. one selected Strong Base checkpoint and machine-verifiable gate evidence.

The public story remains product-facing: M3 is the robust common base from which
recognizable style Bots can safely branch. PFSP and league mechanics are
supporting techniques, not the README headline.

M3 does not implement style rewards, adapters, KL retention, difficulty control,
human evaluation, a distributed learner, or a general-purpose league service.
Those remain M4 and later work.

## 2. Parallel development and the M2 dependency

M3 code and deterministic tests may be developed while the recovered M2
official evaluation runs, but only in the isolated `feat/m3-strong-base`
worktree. The running M2 worktree, process, logs, configs, and evidence targets
must not be modified or monitored by M3 development.

The current `runs/m2/ppo-full/selected.pt` is a **provisional base input**. M3
may use it for unit tests and short development smoke runs. It becomes eligible
for meaningful M3 training only after the M2 evidence audit passes. If M2 fails:

- an integrity or synchronization failure blocks all M3 experiments until M2 is
  repaired;
- a performance-gate failure preserves the M3 infrastructure but requires a
  reviewed base-training decision before long M3 training;
- no partial M2 policy outcome may be used to tune M3.

M2 test rows never become M3 training data. M3 has its own committed train,
validation, and test manifests and opens its test split only after its config,
thresholds, base hash, pool rules, and validation selection are frozen.

## 3. Architecture

M3 is an incremental league overlay around the existing recurrent PPO runtime:

```text
provisional M2 PPO
        |
        v
immutable historical pool ---- validation payoff store
        |                              |
        +-----------> PFSP sampler <---+
                         |
script opponents --------+----> league episode schedule
                                  |
                                  v
                    existing recurrent PPO learner
                                  |
                                  v
                   validation-qualified candidates
                                  |
                                  v
              cross-play + held-out + worst-case gate
                                  |
                                  v
                       selected Strong Base
```

The existing `AsymmetricActorCritic`, recurrent rollout buffer, PPO loss,
checkpoint machinery, synchronous duel runtime, reward ledger, and legal Actor
schema remain authoritative. M3 adds focused modules rather than rewriting the
trainer.

## 4. Opponent contract and fairness boundary

M3 supports two explicit opponent kinds:

- `script`: an existing privileged finite-state Teacher used as a reproducible
  curriculum instrument;
- `checkpoint`: a frozen `RecurrentActor` that receives only its own
  `DuelActorObservation` and recurrent hidden state.

An `OpponentSpec` identifies an opponent by stable ID, kind, source path,
SHA-256, scenario hash, and selection evidence. The collector resolves the spec
at episode start. Script adapters consume `DuelPrivilegedState`; checkpoint
adapters have no privileged-state parameter in their action method. Tests use a
sentinel privileged object and recursive schema inspection to prove a learned
opponent cannot read coordinates, angles, hidden core location, labels, depth,
or future actions.

Historical actors use greedy action selection for both training and payoff
evaluation. This makes every frozen opponent deterministic for a given legal
observation history and avoids adding a second opponent RNG protocol. The PPO
learner remains stochastic during training. Both sides reset recurrent hidden
state at episode boundaries and preserve it across ordinary death/respawn,
matching the M2 learner contract.

All checkpoint loads verify schema version, actor dimensions, scenario hash,
model-state keys, and file SHA-256 before a duel starts. Learned opponents are
loaded read-only and cached on the learner device; a pool of at most 12 actors
is small relative to one 40 GB A100 and avoids per-episode disk loading.

## 5. Historical pool manifest and admission

The pool is a versioned immutable manifest. Each entry contains:

- stable policy ID and checkpoint relative path;
- checkpoint SHA-256, scenario hash, config hash, and source Git commit;
- environment steps and parent/base checkpoint hash;
- validation report path and hash;
- script-pool average and worst-case win rates;
- objective rate and payoff vector against the pool version used at admission;
- `anchor` status and admission reason.

The provisional M2 selected PPO is the permanent anchor. Candidate snapshots are
created at fixed environment-step intervals and evaluated only on the M3
validation split. A candidate is eligible when:

1. checkpoint and fairness audits pass with zero protocol inconsistencies;
2. all validation rows are complete, paired, side-swapped, and finite;
3. its average script-pool win rate is no more than 10 percentage points below
   the active candidate, preventing obviously collapsed policies from entering;
4. it either improves historical worst-case win rate or has a payoff-vector
   L1 distance of at least 0.10 from every active non-anchor entry.

The active pool must contain 8–12 representative policies before the final M3
gate. Until it reaches 12 entries, every eligible candidate is admitted. When
full, the anchor and newest admitted entry are protected; the most redundant
non-anchor entry is replaced only if the new candidate increases minimum
pairwise payoff-vector distance or improves historical worst-case performance.
Every pool version remains recorded, so replacement changes the active view but
does not erase lineage or evidence.

Duplicate hashes, mutable IDs, missing validation evidence, scenario mismatch,
test-derived metrics, or admission computed from incomplete payoff rows are hard
errors.

## 6. League episode schedule and PFSP

M3 introduces a neutral paired `LeagueCase` manifest. A case contains split,
pair index, seed, learner side, declared core-spawn stratum, and route label but
does not bake an opponent identity into the seed. The episode scheduler combines
one case with one `OpponentSpec`. Adjacent host/opponent cases share seed and
opponent, preserving side-swapped comparisons.

The frozen neutral manifests contain 250 train seed-pairs, 50 validation
seed-pairs, 50 standard test seed-pairs, and 50 additional held-out seed-pairs,
with both learner sides materialized for every pair. The held-out manifest is a
second test-only split, not validation data. All four manifests are committed
before meaningful M3 training.

After the historical pool contains at least two entries, the frozen opponent
source mixture is:

- 40% uniform script opponents;
- 50% PFSP historical opponents;
- 10% uniform historical opponents.

For learner validation win rate `p_i` against historical policy `i`, PFSP uses:

```text
w_i = max(0.05, (1 - p_i)^2)
q_i = w_i / sum_j(w_j)
```

The floor prevents policies from disappearing after temporary mastery. The
separate uniform component protects coverage and makes the effective
distribution auditable. Before two historical policies exist, the missing
historical mass is assigned to uniform scripts rather than silently inventing
payoffs.

Sampling is stateless and resume-safe. The source and opponent for a side-swapped
pair are generated from a stable hash of the configured master seed, pair slot,
pool-manifest hash, and payoff-report hash. Resuming at the same episode index
therefore selects the same opponent without serializing hidden sampler state.
Missing, stale, or hash-mismatched payoff data is an error once PFSP is enabled.

Every rollout summary records requested and realized source counts, opponent
counts, PFSP probabilities, pool version, payoff hash, and paired-side balance.

## 7. Training integration and checkpoint semantics

M3 uses a separate `train_league.py` CLI and `configs/m3/league.yaml`; the M2
trainer and frozen config remain untouched. The CLI warm-starts model weights
from the selected M2 PPO checkpoint but creates a fresh optimizer and scheduler
under the M3 config hash. It never pretends an M2 optimizer state belongs to a
different opponent distribution.

The initial frozen training budget is 2,000,000 environment steps with candidate
snapshots every 200,000 steps. A validation pilot may stop the run early for
collapse or integrity failure, but cannot extend the budget or alter admission
rules after M3 test access.

The M3 checkpoint contains the existing model/optimizer/scheduler/RNG payload
plus counters and hashes for:

- provisional/promoted base checkpoint;
- league config and train manifest;
- active pool manifest and payoff report;
- environment steps, updates, episodes, and next paired episode index.

Resume requires exact agreement for every hash and reconciles metrics only up to
the checkpoint's committed counters. Pool changes occur only at declared
validation boundaries. An interrupted rollout cannot partially mutate the pool
or payoff matrix. Checkpoint, summary, pool manifest, and selection writes are
atomic.

Candidate snapshots are generated at fixed intervals. Validation and pool
admission run as separate deterministic commands, so training never reads M3
test rows or silently changes its opponent distribution based on test results.

## 8. Cross-play and validation selection

The payoff evaluator accepts script or checkpoint policies through the same
policy descriptor layer. It runs fixed validation seed-pairs with both sides,
greedy learned actions, identical episode limits, and no reward shaping. It
records win, draw, objective completion, score difference, protocol consistency,
environment retries, and Wilson intervals.

For `N` active historical policies, the published validation cross-play matrix
contains every ordered policy pair, including the diagonal. Side swapping is
mandatory. The raw long-form CSV is authoritative; JSON matrices and the heatmap
are derived artifacts. Missing cells, asymmetric seed sets, duplicate rows,
hash mismatch, NaN/Inf, or inconsistent protocol events invalidate the matrix.
Each unordered policy pair, including self-play diagonal entries, uses the first
five frozen validation seed-pairs with both side assignments. Those paired rows
populate both ordered matrix cells without rerunning an equivalent matchup.

Checkpoint selection uses validation only. Candidates are ranked lexicographically
by:

1. passing every integrity and non-regression constraint;
2. highest historical worst-case win rate;
3. highest script-pool average win rate;
4. highest full-objective rate;
5. earliest environment step, breaking exact ties in favor of lower compute.

The selection report includes every candidate, rejection reason, checkpoint
hash, config/pool/payoff hashes, and the chosen candidate. The final test command
accepts only the selected hash recorded in this committed report.

## 9. Held-out and subtask evidence

M3 reports the section 8.1 capability decomposition from `Plan.md`: goal reach,
pickup, return, valid hit, disengage, and full objective. These metrics are
derived from protocol events and offline privileged evaluation only; privileged
state never becomes an Actor input.

`no_opponent` is an evaluation-only controller that always submits `IDLE`. It is
not added to the training mixture. It measures whether the learned visual policy
can complete the full objective without combat interference.

The current M2 `DuelCase.core_spawn_index` and `route` fields are metadata and do
not force MAP07 geometry or routing. M3 therefore does not claim an unrealized
route intervention. Held-out configurations are frozen as unseen seeds and
side-swaps, stratified after execution by the **actual protocol core coordinates**,
learner side, and opponent kind. The evaluator verifies that all three actual
core locations are represented and reports each stratum separately. Changing
the WAD to force a declared spawn would change the scenario hash and invalidate
the M2 checkpoint, so it is explicitly outside M3.

Disengage evidence uses declared low-health encounters and reports success
descriptively unless a deterministic start-state intervention is available
without changing the scenario hash. It cannot be presented as a controlled
causal test when that intervention is absent.

## 10. Frozen Strong Base gate

Before opening the M3 test split, the following thresholds are copied from
`Plan.md` into a versioned evaluation config and committed:

- average win rate against the five script opponents: at least 70%;
- win rate against every major script opponent: at least 55%;
- no-opponent full-objective completion: at least 90%;
- held-out full-objective completion: at least 80%;
- historical-pool worst-case win rate: strictly higher than the selected M2
  fixed-script-only PPO baseline on the same historical opponents;
- paired bootstrap lower confidence bound for Strong Base minus M2 baseline
  mean score difference against historical opponents: at least zero.

The official gate additionally requires:

- 8–12 active historical policies with unique hashes and valid admission
  evidence;
- a complete cross-play matrix and exact paired seed/side balance;
- zero protocol, synchronization, fairness-schema, or artifact inconsistencies;
- finite confidence intervals and exact checkpoint/config/scenario hashes;
- explicit reporting of all capability metrics and actual core-location strata;
- proof that training, admission, and selection never accessed M3 test rows.

Every threshold is conjunctive. A high aggregate score cannot compensate for a
failed worst-case, held-out, fairness, or integrity gate. If validation shows a
threshold is structurally incompatible with the frozen scenario, exactly one
calibration may be proposed and committed before test access. M2 or M3 test
results can never justify retroactive calibration.

The official sample counts are frozen as follows:

- script-pool gate: all 50 test seed-pairs against each of five scripts, both
  sides, for 500 Strong Base games;
- no-opponent gate: all 50 test seed-pairs, both sides, for 100 games;
- held-out gate: all 50 dedicated held-out seed-pairs balanced across the five
  scripts, both sides, for 100 games;
- historical gate: the first 20 test seed-pairs against every active historical
  policy, both sides, run once for Strong Base and once for the M2 baseline.

Thus the historical comparison contributes 640–960 games for an 8–12 policy
pool, and the complete official suite contains 1,340–1,660 episodes organized
as side-swapped pairs. All
rates report Wilson 95% intervals; score comparisons use a paired bootstrap 95%
interval with 10,000 resamples and committed seed `20260721`.

## 11. Split isolation and anti-leakage controls

M3 manifests are generated once from master seed `20260721`, with unique seeds
across train, validation, standard test, and held-out test. Loaders require an
explicit purpose:

- training and rollout commands accept only `train`;
- payoff, admission, PFSP updates, and checkpoint selection accept only
  `validation`;
- the official gate accepts only `test`/`heldout`, and only the committed
  selected hash.

Every summary records manifest hashes and `test_cases_accessed`. A common audit
recursively checks configs, metrics, pool manifests, and selection reports for
test paths or hashes. Development flags make summaries visibly unofficial and
incapable of passing.

## 12. Failure handling and experiment boundary

All deterministic modules are implemented with strict TDD. CPU unit tests and
short real-duel/GPU smoke runs prove loading, inference, opponent-side
perspective, paired sampling, resume, and evaluation before meaningful compute.

Runtime respawn timeouts retain bounded same-case retries and environment retry
counts. A retry never changes seed, side, opponent, policy, or row identity.
Protocol inconsistency, checkpoint mismatch, non-finite metrics, incomplete
matrix cells, stale payoff data, or dirty official worktree fail closed.

When a meaningful league training, cross-play, or official evaluation needs
`nohup`, its exact launch, PID, log, monitoring, resume, artifact, and success
commands are written to `script.md`. Work then stops and the goal is explicitly
blocked for the user to launch it. Partial logs are health signals only and are
never used for policy comparison or tuning.

## 13. Testing and review gates

The implementation must provide:

- pool schema, hash, duplicate, admission, capacity, and deterministic
  replacement unit tests;
- PFSP formula, floor, mixture, paired-side, resume, missing-payoff, and source
  accounting tests;
- learned-opponent legal observation, recurrent reset, side perspective,
  greedy action, and checkpoint mismatch tests;
- rollout tests covering script and checkpoint opponents without changing PPO
  tensor semantics;
- payoff matrix completeness, pairing, diagonal, confidence, and atomic-write
  tests;
- split-leak, frozen-threshold, no-opponent, held-out-stratum, and Strong Base
  conjunctive-gate tests;
- real duel smoke against one checkpoint opponent with equal tics and complete
  cleanup;
- short GPU forward/rollout/update/resume smoke;
- full pytest, Ruff, deterministic scenario build, M2 regression, and clean
  worktree checks before long experiments.

Each implementation task ends in a focused commit. Existing M2 behavior is
regression-tested; unrelated formatting or refactoring is excluded.

## 14. Public artifacts

M3 publishes compact, evidence-backed artifacts:

- `reports/m3/pool-manifest.json` and admission history;
- raw validation/test episodes and payoff CSV;
- cross-play matrix JSON and heatmap;
- PFSP sampling-distribution and pool-evolution figures;
- selected Strong Base summary, manifest, checkpoint hash, and model card;
- `docs/milestones/m3.md` with commands, limitations, compute, and exact gates;
- a README comparison of M2 PPO versus Strong Base on script, historical,
  worst-case, no-opponent, and held-out dimensions.

Large intermediate checkpoints and raw training runs remain ignored. The final
Strong Base checkpoint is published directly only if repository size policy
allows it; otherwise the repository publishes its hash, release location, and a
reproducible generation command. No result table or figure is hand-authored from
memory: every public number is generated from audited JSON/CSV evidence.

## 15. Staged delivery

M3 proceeds through four reviewable gates:

1. **League substrate:** opponent descriptors, checkpoint opponent, pool
   manifest, PFSP, and deterministic paired schedule.
2. **Training integration:** league collector, warm start, resume, candidate
   snapshots, and short real/GPU smoke.
3. **Evaluation substrate:** cross-play, admission, selection, held-out metrics,
   Strong Base gate, and evidence audit.
4. **Experiments and packaging:** M2 dependency check, long training, matrix,
   official test, plots, model card, milestone page, and README.

Code may advance automatically from one deterministic gate to the next. Any
meaningful `nohup` experiment or proposed change to the approved technical route
requires the agreed user handoff before proceeding.
