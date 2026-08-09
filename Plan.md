# BotColosseo — current release plan

## Outcome

Publish one compact, auditable ViZDoom case that demonstrates:

1. a capable fair-observation Strong Bot;
2. Aggressive, Defensive, and Explorer policies derived from the same Strong
   Actor;
3. randomized-loot evaluation with explicit product and research evidence
   boundaries;
4. representative first-person videos with viewer-only telemetry.

The contribution is the end-to-end behavior-shaping product workflow, not a
claim of novel PPO mathematics.

## Frozen game

- One 75-second 1v1 raid with two neutral extraction zones.
- Extraction opens at 30 seconds and requires a 3-second hold.
- Each player has 100 HP, takes 20 damage per valid hit, starts with 30 rounds,
  and cannot reload or respawn.
- A three-slot backpack stores loot worth 10, 25, or 50; higher-value loot
  deterministically replaces the lowest-value item.
- Death drops all unbanked loot into a collectible corpse cache.
- Kills do not score directly; only extracted value is banked.
- Seven loot items are assigned to 16 collision-free anchors per raid through
  a finite family of 128 randomized layouts.

Any change to map assets, layout generation, rules, Teacher semantics,
evaluation protocol, or metric definitions creates a new experiment identity.

## Current Strong lineage

1. Generate 100,000 train and 20,000 validation transitions with the mask-aware
   privileged Strong Teacher.
2. Train the CNN-GRU Actor with behavioral cloning for 10,000 updates.
3. Warm-start an asymmetric Actor-Critic from the BC Actor.
4. Run conservative recurrent PPO for a 1M-step budget with BC replay,
   reference-policy KL, and scheduled Teacher supervision.
5. Save candidates every 50k steps and select by frozen validation, not by
   training reward or the final checkpoint.
6. Freeze the selected 950k Strong checkpoint for every style adapter.

The framework implements historical-opponent sampling and PFSP. The released
Strong configuration sets `history_probability: 0.0`; PFSP is therefore not
part of the empirical claim for this checkpoint.

## Current style lineage

Freeze the selected Strong Actor and train a bounded residual logit adapter:

```text
style logits = strong logits + max_delta * tanh(delta(features))
```

Style PPO combines task reward with:

- opportunity-conditioned utility, so shaping activates only when a meaningful
  style decision exists;
- finite-state potential-based reward shaping for complete behavior chains;
- a preferred-action margin inside opportunities;
- stronger KL to Strong outside opportunities;
- residual-magnitude regularization.

Aggressive converts real combat into extracted value. Defensive disengages to
preserve carried value under risk. Explorer visits useful regions, improves
loot, and converts that exploration into extraction.

## Observation integrity

The deployed Actor receives only:

- an 84×84 first-person grayscale frame;
- own public health, ammunition, backpack, banked value, extraction state,
  remaining time, and previous action.

Opponent state, world coordinates, full loot state, labels, automap, and viewer
telemetry are excluded from Actor inputs. Those fields may be used only by the
training Critic, Teacher, reward shaping, offline evaluation, and video overlay.

## Frozen evaluation

Protocol: `configs/extraction/randomized/evaluation.yaml`.

| Split | Episodes per policy | Layout family | Use |
|---|---:|---|---|
| validation | 240 | randomized | candidate selection and confirmation |
| heldout | 120 | randomized, new seeds | generalization check |
| solo | 40 | randomized, idle opponent | task-completion check |
| official test | 400 | sealed randomized cases | deferred research release |

Validation and heldout share the same map geometry and finite layout family but
use disjoint seed ranges and permutations.

Strict Strong gates require the frozen capability thresholds, positive value
advantage, and zero protocol, truncation, leak, or test-access errors. Strict
style gates require capability retention plus statistically positive style
direction. Failed strict gates are preserved and disclosed; thresholds are not
relaxed after seeing results.

## Current evidence

- Strong is admitted as a product Showcase checkpoint. On the frozen
  240-episode randomized confirmation it reaches 83.3% extraction, 56.7% win
  rate, and 39.10 mean banked value; heldout extraction is 85.8%.
- Aggressive, Defensive, and Explorer are representative validation case
  studies with explicit aggregate-gate disclosures. They demonstrate complete
  engine-recorded causal chains, not distribution-level style improvement.
- Showcase videos are fresh deterministic renders bound to protocol, seed,
  learner side, opponent, layout, and checkpoint. They are not claimed to be
  frame-identical historical evaluation replays.
- Candidate selection records `test_cases_accessed: false`. The single-use
  official-test protocol remains unrun.

## Release gates

| Gate | Evidence | Status |
|---|---|---|
| Scenario | deterministic build, synchronized reset, randomized layouts | PASS |
| Engineering | unit, portable integration, and artifact audits | REQUIRED |
| Strong product evidence | frozen randomized confirmation and heldout report | PASS |
| Strict Strong research gate | all frozen capability thresholds | FAIL, disclosed |
| Style product evidence | four audited videos and complete causal chains | PASS |
| Strict aggregate style gates | distribution-level direction and retention | FAIL, disclosed |
| Official test | sealed 400 episodes per policy | DEFERRED |

The public release is a completed product Showcase with honest evidence tiers.
It is not presented as a successful benchmark or a completed ablation study.
