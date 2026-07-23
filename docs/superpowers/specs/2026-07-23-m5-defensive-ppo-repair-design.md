# M5 Defensive Closed-Loop PPO Repair Design

**Status:** Approved through delegated recommendation authority on 2026-07-23.

## 1. Evidence-driven reason for the repair

The first Defensive route completed its engineering workflow but failed the
frozen style gate. The alpha-0.25 candidate retained 96.85% of Strong Base
performance over 200 paired episodes, while its protective-presence shift was
only `+0.0066` with a 95% interval of `[-0.0468, 0.0679]`.

This rules out a simple small-sample explanation. Success-filtered distillation
preserved task skill but did not create stable closed-loop protection. The
repair therefore adds state-distribution feedback instead of changing the
metric, weakening the gate, or collecting the same demonstrations again.

## 2. Chosen route

Warm-start the existing Defensive alpha-0.25 checkpoint and run one
200,000-environment-step PPO phase. Keep the fair-observation CNN and GRU
frozen; train only the residual adapter, copied policy head, and critic. Retain
the style-to-base KL penalty.

The fixed initial configuration is:

- environment steps: 200,000;
- learning rate: `1e-4`;
- style reward scale: `0.30`;
- style KL coefficient: `0.10`;
- one candidate boundary at 200,000 steps;
- existing M3 train cases, historical pool, PFSP schedule, task reward, and
  recurrent PPO settings.

There is no post-result reward grid. If this route fails, its evidence is
preserved before any new mechanism is proposed.

## 3. Privileged training-only reward boundary

The Actor input remains unchanged and public-only. The rollout collector passes
the step-before and step-after `DuelPrivilegedState` only to a training reward
ledger. Those states are already available to the asymmetric critic and
Teacher infrastructure; they are never appended to actor frames, scalars,
previous actions, masks, or recurrent state.

The generic style reward interface gains explicit `state_before` and
`state_after` keyword arguments. Aggressive ignores them. Defensive requires
them and fails closed when they are absent.

## 4. Defensive reward v2

All terms are side-normalized and reuse the frozen risk, protective-zone, and
defensive-half predicates.

| Component | Event | Value | Per-episode cap |
|---|---|---:|---:|
| `risk_presence` | at risk and learner ends the decision in the protective zone | +0.01 | 40 |
| `protective_entry` | at risk and learner enters the protective zone | +0.05 | 8 |
| `carrier_denial` | opponent carried before the step, then dropped after a learner hit | +0.20 | 6 |
| `defensive_recovery` | loose core in defensive half becomes learner-carried | +0.15 | 6 |
| `risk_resolution` | risk becomes false without an opponent score | +0.10 | 6 |
| `unnecessary_guard` | learner ends in protective zone when no risk exists | -0.02 | 30 |
| `risk_concession` | opponent scores while risk was active | -0.20 | 5 |

The style scale multiplies the capped component values. Existing task reward
remains active.

Threat prolongation cannot generate unbounded presence reward. A concession is
penalized, resolving risk is rewarded, no-risk guarding is penalized, and the
unchanged objective-retention gate prevents protection from replacing scoring.

## 5. Warm-start and checkpoint identity

The training CLI accepts `style=defensive` and a hash-bound Defensive
style-interpolation checkpoint. Loading validates:

- style is `defensive`;
- base and scenario hashes match the requested Strong Base;
- checkpoint SHA-256 matches the CLI identity;
- interpolation metadata are valid;
- test data were not accessed.

The training summary records the warm-start hash, reward components, KL stops,
environment steps, scenario, pool/payoff identities, and
`test_cases_accessed: false`.

## 6. Evaluation and promotion

The 200k checkpoint receives the existing Defensive evaluator unchanged.

1. A 20-episode paired validation smoke must pass all eight current gates.
2. Only a smoke-passing checkpoint may enter the 200-episode formal validation.
3. Formal evidence must pass the unchanged
   `paired_cluster_bootstrap_pooled_ratio_v1` protective-presence estimator,
   skill-retention, per-opponent retention, denial/recovery, unnecessary-guard,
   objective, protocol, and zero-test-access gates.
4. The evidence audit binds the PPO checkpoint and upstream warm-start hash.

No alpha expansion, confidence-interval change, threshold waiver, or
hand-selected video can promote a failed result.

## 7. Implementation boundary

The repair adds:

- `DefensiveRewardConfig` and `DefensiveRewardLedger`;
- before/after state delivery to style reward shapers;
- generic `aggressive|defensive` style validation in `train_league`;
- Defensive style-interpolation warm-start loading;
- a 200k Defensive PPO config and one-command production pipeline;
- reward, public-boundary, checkpoint-identity, CLI, and anti-hacking tests.

It does not change the scenario, data split, evaluator, Strong Base, historical
pool, or Defensive distillation artifacts.

## 8. Verification

Unit tests cover every reward positive case, negative case, cap, reset,
side-mirroring, missing privileged state, threat-prolongation bound, and no-risk
guard penalty. Collector tests prove privileged states reach only the reward
ledger. CLI tests prove Defensive warm starts are accepted while wrong style,
base, scenario, or hash fail. A 2,000-step real CUDA/ViZDoom smoke precedes the
200k production run.
