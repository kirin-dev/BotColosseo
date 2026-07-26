# Crystal Run: Extraction v3 — execution plan

## Outcome

Publish one clean, auditable ViZDoom case that demonstrates:

1. a capable fair-observation Strong Bot;
2. three learned and visibly different styles derived from that same Strong
   Actor;
3. paired capability-retention evidence;
4. representative validation videos with viewer-only telemetry;
5. a frozen, single-use official-test boundary.

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
4. Train recurrent PPO for 2,000,000 environment steps.
5. Mix scripted opponents with historical Strong checkpoints using
   deterministic lightweight PFSP.
6. Rank candidates using validation only; evaluate heldout only for the
   validation-selected candidate.

### Styles

Freeze the selected Strong Actor and train a bounded residual delta-logit
adapter for each style:

```text
style logits = strong logits + max_delta * tanh(delta(features))
```

Optimize:

```text
task reward + style reward - beta * KL(style || Strong)
            - rho * residual magnitude
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

Training-only privileged Critic/reward/evaluation:

- both poses and health values;
- both inventories and banked values;
- cache and world-loot state.

No privileged field may enter the Actor, checkpoint opponent, selected public
policy, or video policy path.

## Frozen evaluation

Protocol: `configs/extraction/evaluation.yaml`.

| Split | Episodes per policy | Layout | Use |
|---|---:|---|---|
| validation | 240 | base | candidate ranking and style gates |
| heldout | 120 | heldout-a | Strong generalization gate |
| official test | 400 | heldout-a | one frozen report after release |

Strong gate:

- prevent opponent extraction ≥90%;
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

## Release

The release manifest binds:

- scenario and evaluation-protocol hashes;
- selected Strong and three style checkpoint hashes;
- validation and heldout selection reports;
- source Git commit;
- official-test lock and receipt;
- validation-only video evidence and media hashes.

Videos are selected from validation, never test. The overlay may display both
HP bars, damage, cache transfer, backpack, and extraction progress for viewers,
but none of these privileged overlay values enter the policy.

## Milestones

| Gate | Evidence | Status |
|---|---|---|
| G0 scenario and layouts | reproducible WAD, real two-player reset, human layout approval | PASS |
| G1 engineering | full unit suite, real Teacher→BC→PPO→style→evaluation preflight | PASS |
| G2 Strong | full demonstrations, BC, PPO, validation selection, heldout gate | PENDING |
| G3 styles | Aggressive, Defensive, Explorer training and paired gates | PENDING |
| G4 release | frozen manifest, one official test, representative videos | PENDING |
| G5 public cleanup | v3-only public narrative and audited artifacts | IN PROGRESS |

The project is complete only when G0–G5 are evidenced and no required work
remains.
