# M5 Defensive: completed experiment, failed style gate

## Status

The Defensive engineering path is complete through data generation,
frozen-base distillation, fixed-alpha interpolation, paired validation, and
hash-bound evidence. The current Defensive model is **not** a successful style
checkpoint: no candidate passed the predefined protective-presence gate.

This record preserves the negative result so that the public project
distinguishes a reproducible workflow from a benchmark success.

## Frozen route

- Strong Base:
  `runs/m3/league-full/candidate-boundary-0200000.pt`
- Training data: 50,000 risk-conditioned transitions in five hash-bound shards
- Training: 1,000 frozen-base style-adapter distillation updates
- Candidate grid: `alpha = 0.25, 0.50, 0.75`
- Smoke protocol: 20 paired validation episodes per candidate
- Diagnostic protocol: 200 paired validation episodes for the most promising
  `alpha = 0.25` candidate
- Estimator: `paired_cluster_bootstrap_pooled_ratio_v1`
- Test-case access: false

The production data manifest retained 95 of the predefined 100
denial/recovery windows. A narrow, manifest-specific waiver admitted only that
artifact to distillation; it did not relax any downstream style, capability, or
integrity threshold.

## Candidate evidence

| Alpha | Skill retention | Protective-presence delta | 95% interval | Result |
|---:|---:|---:|---:|---|
| 0.25 | 1.000 | +0.1396 | `[-0.0130, 0.2132]` | Failed style CI |
| 0.50 | 0.975 | -0.0655 | `[-0.3649, 0.0397]` | Failed style and recovery |
| 0.75 | 0.975 | +0.0139 | `[-0.2285, 0.1198]` | Failed style CI |

The alpha-0.25 smoke suggested a possible positive shift but was not
statistically stable. Its 200-episode diagnostic retained 96.85% of Strong Base
performance, while protective presence moved by only `+0.0066` with a 95%
paired interval of `[-0.0468, 0.0679]`. Seven of eight gates passed; the
protective-presence shift did not.

## Interpretation

The result rules out sampling noise as the main explanation for the alpha-0.25
smoke. Success-filtered behavior cloning preserved task skill but did not
produce a stable closed-loop Defensive behavior. A future repair should add a
closed-loop reinforcement signal or revise the behavior target; it should not
weaken the frozen metric or relabel this checkpoint as passed.

## Evidence

- [Data waiver](../../reports/m5/defensive/data-waiver.json)
- [Fixed-grid selection](../../reports/m5/defensive/selection.json)
- [Alpha-0.25 smoke](../../reports/m5/defensive/smoke/alpha-025/summary.json)
- [Alpha-0.50 smoke](../../reports/m5/defensive/smoke/alpha-050/summary.json)
- [Alpha-0.75 smoke](../../reports/m5/defensive/smoke/alpha-075/summary.json)
- [200-episode diagnostic](../../reports/m5/defensive/diagnostic-alpha-025-formal/summary.json)
- [Diagnostic manifest](../../reports/m5/defensive/diagnostic-alpha-025-formal/manifest.json)

The evaluation ledger and manifests are committed with the report so their
hashes remain independently checkable.

## Closed-loop PPO repair result

The first closed-loop repair then warm-started alpha 0.25 and completed
200,000 real environment steps with zero KL early stops. It preserved task
skill but did not repair the style mechanism:

| Frozen 20-episode smoke | Result |
|---|---:|
| Skill Retention | 89.74% |
| Protective-presence delta | -0.0875 |
| 95% interval | `[-0.3345, 0.0295]` |
| Protocol inconsistencies / retries | 0 / 0 |

The run failed protective-presence shift, denial/recovery improvement, and
per-opponent retention. Its training ledger never recorded a carrier-denial or
defensive-recovery reward. Scaled unnecessary-guard and concession penalties
also exceeded all positive defensive rewards combined, making risk avoidance
the easier learned behavior.

The failed smoke is retained under
[`reports/m5/defensive/ppo-repair/smoke/`](../../reports/m5/defensive/ppo-repair/smoke/).
The owner-approved V2 route adds masked on-policy Protective Teacher
regularization and a bounded 50k/100k budget; it does not change this evaluator.

## Teacher-assisted V2 result

V2 passed both its 2,000-step real CUDA preflight and 50,000-step training
audit. The production ledger contained 25,496 risk-supervised tokens and,
unlike V1, recorded both carrier-denial and defensive-recovery rewards. The
hash-bound 20-episode smoke was complete and protocol-clean:

| Frozen 50k decision input | Result |
|---|---:|
| Skill Retention | 92.31% |
| Protective-presence delta | -0.0183 |
| 95% interval | `[-0.2361, 0.3026]` |
| Denial/recovery gate | Pass |
| Protocol inconsistencies / retries | 0 / 0 |

The primary point estimate still had the wrong sign, while unnecessary guard
also increased. The pre-approved decision rule therefore produced
`stop_50k`; V2 was not extended to 100k. See the
[decision record](../../reports/m5/v2/defensive/decision-050000.json) and
[smoke evidence](../../reports/m5/v2/defensive/smoke-050000/summary.json).
