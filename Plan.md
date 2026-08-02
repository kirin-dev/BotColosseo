# BotColosseo v3 — execution plan

## Outcome

Publish one clean, auditable ViZDoom case that demonstrates:

1. a capable fair-observation Strong Bot;
2. three learned and visibly different styles derived from that same Strong
   Actor;
3. paired validation evidence with explicit research/product evidence tiers;
4. representative validation videos with viewer-only telemetry;
5. a clean public Showcase that does not imply an official-test result.

The project contribution is the complete behavior-shaping product workflow,
not a claim of novel PPO mathematics.

## Frozen game

- 1v1, 75 seconds, two neutral extraction zones.
- Extraction opens at 30 seconds and requires a 3-second hold.
- 100 HP, fixed 20 damage, 30 ammunition, no reload, no respawn.
- Three-slot backpack; values 10, 25, 50; deterministic replacement.
- Death creates a corpse cache.
- Only extracted value scores.
- `base` layout for train/validation; approved `heldout-a` for heldout/test.

Any further map geometry, rule, damage, inventory, or extraction change
invalidates the scenario hash and requires a new human review before training.

## Method

### Strong

1. Generate 100,000 fair-observation Strong Teacher transitions and 20,000
   validation transitions.
2. Train a CNN-GRU Actor with behavioral cloning for 10,000 updates.
3. Warm-start an asymmetric Actor-Critic from the BC Actor.
4. Train recurrent PPO through the frozen 2,000,000-step candidate horizon,
   saving resumable checkpoints about every 200k steps; select by validation
   rather than defaulting to the final checkpoint.
5. Mix scripted opponents with historical Strong checkpoints using
   deterministic lightweight PFSP.
6. Rank candidates using validation only; evaluate heldout only for the
   validation-selected candidate.

Closeout auditing fixed PFSP draw bookkeeping after the frozen Strong
checkpoint had been trained. Its independent validation/heldout evidence
remains the current product result; a causal PFSP-gain claim requires retraining.

### Styles

Freeze the selected Strong Actor and train a bounded residual delta-logit
adapter for each style:

```text
style logits = strong logits + max_delta * tanh(delta(features))
```

Build the PPO return from:

```text
r_t = environment task reward + task shaping + style shaping
```

Then minimize:

```text
L_style = L_PPO(r_t)
        + beta * KL(style || Strong)
        + rho * squared residual magnitude
```

Aggressive rewards useful hits and kill-cache-extraction conversion.
Defensive rewards meaningful risk disengagement and penalizes empty camping.
Explorer rewards real new loot regions, backpack upgrades, and extraction
conversion rather than raw movement.

Aggressive is the first vertical slice. Defensive and Explorer begin only
after the Aggressive path is engineering-valid.

## Observation integrity

Public Actor input:

- first-person grayscale pixels;
- own public health, ammunition, backpack, banked value, extraction state,
  remaining time, previous action.

Privileged training and offline-evaluation support:

- the asymmetric training Critic and reward shaping may use both poses, health
  values, inventories, banked values, cache state, and world-loot state;
- offline evaluation and viewer telemetry may use the same state to score and
  explain behavior.

No privileged field may enter the Actor, checkpoint opponent, selected public
policy, or video policy path.

## Frozen evaluation

Protocol: `configs/extraction/evaluation.yaml`.

| Split | Episodes per policy | Layout | Use |
|---|---:|---|---|
| validation | 240 | base | candidate ranking and style gates |
| heldout | 120 | heldout-a | Strong generalization gate |
| official test | 400 | heldout-a | deferred strict research release only |

Strong gate:

- solo extraction against an idle opponent ≥90%;
- average scripted win rate ≥70%;
- every scripted-opponent win rate ≥55%;
- validation extraction rate ≥75%;
- heldout extraction rate ≥70%;
- positive mean extracted-value advantage;
- zero protocol, truncation, leak, and test-access errors.

Style gate:

- paired Strong success retention ≥85%;
- extraction delta ≥−10 percentage points;
- mean extracted value ≥85% of Strong;
- paired style direction >0;
- 95% paired confidence lower bound >0;
- zero integrity errors.

If a frozen gate fails, preserve the report and describe it honestly. Do not
change thresholds after observing test results.

Product Showcase evidence is separately tiered:

- `research_selection`: all frozen validation and heldout gates pass;
- `directional_showcase`: approved product-direction evidence with disclosed
  research failures;
- `validation_demonstration`: paired validation direction, capability,
  anti-hacking, and protocol checks pass, while heldout failures are disclosed.
- `representative_case_demonstration`: a post-validation product case study
  with aggregate failures disclosed. It proves only the selected causal chain,
  never distribution-level style improvement. For Defensive, product safety is
  based on own extraction (delta at least -10 points), mean banked value (at
  least 80% of Strong), bounded timeout regression, and a complete disengage-to-
  extraction case; opponent denial is not a product objective.

The last three tiers are validation-media only and cannot enter the strict
official-test release.

## Release

The product Showcase manifest binds:

- scenario and evaluation-protocol hashes;
- selected Strong and three style checkpoint hashes;
- validation and heldout selection reports;
- validation-only video evidence and media hashes.

Videos are selected from validation, never test. The overlay may display both
HP bars, damage, cache transfer, backpack, and extraction progress for viewers,
but none of these privileged overlay values enter the policy.

The repository revision provides source provenance separately. The current
product Showcase manifest does not embed a Git commit.

## Milestones

| Gate | Evidence | Status |
|---|---|---|
| G0 scenario and layouts | reproducible WAD, real two-player reset, human layout approval | PASS |
| G1 engineering | full unit suite, real Teacher→BC→PPO→style→evaluation preflight | PASS |
| G2 Strong | full demonstrations, BC, PPO, validation selection; heldout research gate failed | PRODUCT PASS / RESEARCH FAIL |
| G3 product styles | three learned adapters with explicit evidence tiers | PASS |
| G4 Showcase | contrastive validation cases, four audited videos, public board | PASS |
| G5 public cleanup | v3-only public narrative and current-run audited artifacts | IN PROGRESS |

The current product Showcase is complete through G4. G5 closes after the
current-run assets, honest evidence tiers, and failed research gates are
published on `main`. A strict all-style research release and the single-use
official test remain clearly labeled future work.
