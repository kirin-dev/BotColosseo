# M5 Explorer: successful offline route learning, failed closed-loop style gate

## Status

The Explorer data, distillation, interpolation, paired smoke evaluation, and
deterministic selection workflow completed. No candidate passed the frozen
closed-loop Explorer gate, so the current checkpoint is not presented as a
successful Explorer Bot.

## What passed

The production data gate passed every requirement:

- 50,000 public-observation train transitions in five hash-bound shards;
- 5,422 route-supervised transitions;
- retained successful windows: 15 direct-upper, 15 direct-lower, 11 flank;
- retained route proportions: 36.6%, 36.6%, and 26.8%;
- one-third Strong Base context replay;
- all five opponents and both learner sides;
- zero test-case access.

The initial `beta_kl=0.05` distillation learned route targets but caused 16.40%
Base-context drift. A frozen KL calibration used the same data and thresholds:

| KL coefficient | Route-target agreement delta | Base-context drift | Result |
|---:|---:|---:|---|
| 0.05 | +48.79% | 16.40% | Failed retention |
| 0.10 | +43.54% | 13.30% | Failed retention |
| 0.20 | +34.96% | 9.69% | Passed offline gate |

The minimum passing choice, `beta_kl=0.20`, was promoted before closed-loop
evaluation.

## What failed

Each fixed interpolation candidate received the same 20-episode paired
validation smoke:

| Alpha | Skill retention | Route-entropy delta | Flank completions | Result |
|---:|---:|---:|---:|---|
| 0.25 | 97.44% | -0.1262 | 0 | Failed |
| 0.50 | 82.50% | -0.0631 | 0 | Failed |
| 0.75 | 102.56% | 0.0000 | 0 | Failed |

All candidates were protocol-clean. The decisive failure is behavioral: none
completed a flank route, so route coverage, flank improvement, and positive
route-entropy shift all failed. Alpha 0.50 also failed capability retention.

## Interpretation

The Teacher and offline learner captured distinguishable route actions, but
one-step imitation did not sustain the multi-decision flank behavior under the
policy's own state distribution. The appropriate next mechanism is a
closed-loop route-completion objective or hierarchical route commitment—not a
weaker entropy gate or a hand-picked video.

## Evidence

- [Production data manifest](../../reports/m5/explorer/data-manifest.json)
- [KL calibration](../../reports/m5/explorer/kl-calibration.json)
- [Passing distillation summary](../../reports/m5/explorer/distillation-summary.json)
- [Fixed-grid selection](../../reports/m5/explorer/selection.json)
- [Alpha-0.25 smoke](../../reports/m5/explorer/smoke/alpha-025/summary.json)
- [Alpha-0.50 smoke](../../reports/m5/explorer/smoke/alpha-050/summary.json)
- [Alpha-0.75 smoke](../../reports/m5/explorer/smoke/alpha-075/summary.json)

The smoke ledgers, summaries, run identities, and manifests are committed with
this report.
