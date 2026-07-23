# M5 Explorer Closed-Loop PPO Repair Design

**Status:** Approved through delegated recommendation authority on 2026-07-23.

## 1. Evidence-driven reason

Explorer passed its 50,000-transition data gate and its KL-calibrated offline
distillation gate. In fixed 20-episode validation smokes, however, all three
interpolation candidates completed zero flank routes. Route-entropy deltas were
negative or zero even when skill retention remained high.

The mechanism gap is temporal: one-step imitation learned route actions but did
not sustain a multi-decision route commitment under the policy's own state
distribution. The repair adds closed-loop route-completion feedback without
weakening route coverage, entropy, efficiency, or capability gates.

## 2. Chosen route

Warm-start the alpha-0.25 Explorer checkpoint and train one 200,000-step PPO
run. Keep the CNN and GRU frozen; update only the residual adapter, copied
policy head, and critic. Preserve the task reward and style-to-base KL.

Initial fixed settings:

- environment steps: 200,000;
- learning rate: `1e-4`;
- style reward scale: `0.30`;
- style KL coefficient: `0.10`;
- existing M3 train cases, PFSP schedule, recurrent PPO settings, and one 200k
  candidate boundary.

No validation-observed reward grid is added.

## 3. Route target

The target remains the approved legal-history cycle:

```text
(learner own score - episode initial learner own score) mod 3
→ direct_upper, direct_lower, flank
```

The route ledger lazily records the initial learner score on its first step and
uses only within-episode score change to advance the target. Route identity is
training/evaluation metadata, not an Actor input.

## 4. Explorer reward v2

Rewards apply only while the learner carries the core.

| Component | Event | Value | Per-episode cap |
|---|---|---:|---:|
| `target_region` | first visit to the target route's required region during the current carry | +0.05 | 12 |
| `target_route_score` | learner scores after completing the target route signature | +0.25 | 5 |
| `novel_carry_region` | first visit to any region during the current carry | +0.01 | 24 |
| `carry_stall` | learner remains in the same region for more than 12 carrying decisions | -0.01 | 30 |

Direct-upper requires `upper_route`; direct-lower requires `lower_route`; flank
requires both `flank_west` and `flank_east`. A target milestone is rewarded once
per carry. Pickup starts a fresh carry path; drop or death clears it. Score
checks the completed carry path, then advances the target through public score
history.

Wrong-route scoring is not punished because it still completes the task. It
simply receives no target-route completion bonus. Non-carrying region visits,
turning, and action frequency receive no positive style reward, preventing
reward for purposeless wandering.

## 5. Boundary and identity

The rollout collector already supplies before/after privileged state only to
the training reward ledger. Explorer uses learner region, carrier, and score
from that boundary. Actor tensors remain unchanged.

Warm-start validation binds style, alpha, base, scenario, checkpoint hash, and
the zero-test-access interpolation report. Training records config, pool,
payoff, warm-start, scenario, train-manifest, candidate, reward-component, and
test-access identities.

## 6. Promotion

The single 200k candidate receives the existing Explorer evaluator unchanged:

1. 20 paired validation episodes must pass all route, retention, efficiency,
   and protocol gates;
2. only a smoke-passing candidate receives the 200-episode formal evaluation;
3. the existing paired normalized-entropy estimator and all hard gates remain
   frozen;
4. an independent audit binds the PPO checkpoint to its warm start and formal
   ledger.

No runtime action override, hand-selected route, alpha expansion, or
post-result gate change is permitted.

## 7. Deliverables and verification

Implementation adds an `ExplorerRewardLedger`, `style=explorer` training
support, a frozen config, one-command CUDA-smoke/production/evaluation
pipeline, and PPO evidence audit.

Tests cover target cycling, upper/lower/flank milestones, completion reward,
wrong-route non-reward, non-carry no-reward, drop reset, stall cap, side
handling, missing state, public Actor isolation, warm-start identity, and
tampered evidence. A 2,000-step real CUDA smoke must show non-empty Explorer
reward components before the 200k run.
