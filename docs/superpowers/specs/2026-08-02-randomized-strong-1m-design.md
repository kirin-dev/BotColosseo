# Randomized Strong 1M Training Design

## Objective

Train a stronger fair-observation Strong Bot under randomized loot placement
with a 1,000,000-environment-step PPO budget. The new run must preserve the
existing 200k checkpoint and public Showcase, remain recoverable, and be
selected only from validation evidence produced after training finishes.

This design extends the finite domain-randomization environment already in the
repository: seven loot items are assigned to 16 safe anchors through 128
deterministic collision-free permutations. It does not claim continuous
coordinate randomization.

## Scope

Included:

- a mask-aware privileged Strong Teacher;
- a three-stage layout curriculum;
- online Teacher-loss annealing;
- delayed low-learning-rate unfreezing of the final convolutional layer;
- a 1M learning-rate and reward-shaping schedule;
- resumable 50k checkpoints without online evaluation;
- post-training 32/120/240-episode validation selection.

Excluded:

- changes to map geometry, combat, inventory, extraction, or scoring rules;
- opponent-difficulty curriculum;
- full visual-backbone unfreezing;
- style-bot retraining;
- official-test access;
- replacement of the public Showcase Strong before the new candidate passes.

## Training curriculum

The curriculum changes only the number of eligible loot-layout variants. Both
learner sides and every configured opponent class remain stratified throughout.

| Phase | Environment steps | Eligible variants | Visual backbone |
|---|---:|---:|---|
| 1 | 0–100k | 16 | frozen |
| 2 | 100k–300k | 32 | frozen |
| 3 | 300k–600k | 128 | frozen |
| 4 | 600k–1M | 128 | final convolutional layer unfrozen |

The scheduler derives its phase from committed `environment_steps`, not wall
time or episode count. Resuming a checkpoint therefore reconstructs the same
eligible case set. Sampling is balanced across learner side and opponent type;
it must not obtain a curriculum by taking a raw prefix of the case manifest.

## Mask-aware Strong Teacher

The privileged Teacher may reconstruct loot coordinates from the layout seed
and read `world_loot_mask`, but those values remain unavailable to the Actor.
At every decision:

1. Continue toward the cached target while its mask bit remains active.
2. Immediately invalidate a target removed by either player.
3. With a free slot, select an active useful item deterministically.
4. With a full backpack, ignore items that cannot improve its minimum slot.
5. When no useful target remains or time is short, extract carried value.
6. At 85 carried value, transition directly to extraction.
7. Preserve the existing bounded combat behavior.

Target selection must be deterministic and stable: keep the current target
until it disappears, becomes useless, or is reached. This prevents oscillation
between equally valued anchors. No Teacher-only feature is added to Actor
observations.

## Optimization schedules

### Teacher auxiliary loss

| Environment steps | Coefficient |
|---|---:|
| 0–100k | 1.0 |
| 100k–600k | linear 1.0 → 0.2 |
| 600k–1M | 0.2 |

This is online privileged Teacher supervision, not an additional BC dataset.
It anchors early task skill while allowing later PPO updates to depart from an
imperfect Teacher.

### Reward shaping

Existing task shaping decays linearly to zero at 800k. The final 200k therefore
optimizes the primary environment objective without dense shaping. No reward is
added for forced kills or preventing opponent extraction; denial remains an
evaluation metric only.

### Learning rates and parameter groups

- Scalar encoder, GRU, policy head, and Critic remain trainable from step zero.
  They start at `1e-5` and follow a 1M cosine schedule with a nonzero `1e-6`
  floor.
- "Frozen visual backbone" refers only to `actor.visual_encoder`; it does not
  freeze the scalar encoder or GRU as the current helper does.
- Exactly the third and final `Conv2d` module (`visual_encoder[4]`) remains
  present in the optimizer with learning rate zero through 600k.
- At 600k its learning rate warms up over 20k steps to `5e-7`, then decays to
  `1e-7` at 1M.
- The earlier convolutional layers and the visual projection `Linear` layer
  stay frozen for the entire run.

Optimizer groups, scheduler state, curriculum state, and committed counters are
checkpointed. Resume must not silently restart warm-up or change the eligible
layout set.

## Checkpointing and execution

- Use a new output directory; preserve the completed 200k run.
- Save candidates every 50k environment steps, yielding 20 candidates.
- `latest.pt` is a recovery artifact, not an automatically selected policy.
- Do not run ViZDoom validation during PPO training.
- Launch through `nohup` on GPU 0 with an explicit log and PID record.
- Audit startup after the first rollout and again near 50k, then allow the run
  to finish without frequent polling.
- Expected PPO duration from current throughput is approximately 5–7 hours.

Startup audit must prove the randomized scenario hash, BC checkpoint hash,
curriculum phase, Teacher coefficient, parameter-group learning rates, fair
Actor observation, and `test_cases_accessed=false`.

## Post-training selection

Selection uses validation only and proceeds as a funnel.

### Screening

Evaluate all 20 checkpoints on the same 32 paired randomized-validation
episodes. Rank lexicographically by:

1. extraction rate;
2. win rate;
3. mean extracted-value advantage;
4. lower death rate.

Retain four candidates. Do not use an arbitrary weighted score.

### Expanded validation

Evaluate the four candidates on 120 balanced randomized episodes using paired
bootstrap intervals, then retain two.

### Final comparison

Evaluate the two finalists, the current Randomized Strong 200k, and the original
fixed-layout Strong on the same 240 randomized episodes. Evaluate each finalist
on an additional 120 base-layout and 120 heldout-a episodes.

## Promotion gates

A finalist may replace the current Randomized Strong only if:

- protocol inconsistencies equal zero;
- Actor privilege violations equal zero;
- `test_cases_accessed=false` throughout;
- randomized extraction improves by at least 5 percentage points over the
  current 200k policy on the same 240 episodes;
- win rate does not regress by more than 2.5 points;
- death rate does not worsen by more than 5 points;
- base and heldout-a extraction each regress by no more than 5 points.

Aspirational targets are at least 80% randomized extraction, at least 60% win
rate, and at most 25% death rate. These targets are not grounds for rewriting or
hiding a failed result. If no candidate passes, retain the current 200k policy
and report the 1M run as unsuccessful.

## Failure handling

- Reject resume when scenario, BC, config, case-manifest, optimizer-group, or
  curriculum identity differs.
- Reconcile metrics only to the committed checkpoint boundary after interruption.
- Treat nonfinite loss, invalid curriculum phase, learning-rate mismatch, test
  access, or protocol inconsistency as a hard failure.
- A single bad candidate does not terminate the run; a training-process failure
  does, leaving the latest committed checkpoint recoverable.

## Verification

Before launch:

- unit-test curriculum boundaries, stratified sampling, deterministic resume,
  Teacher target invalidation, backpack usefulness, loss coefficients, visual
  unfreeze boundaries, and learning-rate schedules;
- rebuild the randomized WAD and confirm its hash is unchanged;
- run a short CUDA smoke through curriculum transitions using compressed test
  boundaries;
- confirm the existing 200k artifacts and public Showcase are untouched.

After training:

- audit all 20 checkpoint hashes and counters;
- verify the final training summary reports exactly 1M committed steps;
- execute the validation funnel without test access;
- produce one machine-readable selection report containing every candidate,
  rejected gate, paired comparison, and selected checkpoint hash.
