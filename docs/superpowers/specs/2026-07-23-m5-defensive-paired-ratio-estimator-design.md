# M5 Defensive Paired Ratio Estimator Design

Date: 2026-07-23  
Status: approved by the project owner

## Problem

The Defensive validation gate compares protective presence during verified risk
states. The current estimator first computes a rate for every episode and then
equally averages paired episode-rate differences:

```text
mean(defensive_presence / defensive_risk
     - base_presence / base_risk)
```

An episode with zero risk decisions receives a rate of zero. This treats the
absence of a risk opportunity as failed defensive behavior. It is especially
unstable when the two policies induce different trajectories and therefore
different risk denominators.

The alpha 0.25 smoke ledger demonstrates the defect. Its pooled Defensive and
Strong Base rates are 0.4925 and 0.3529, respectively, but the current
episode-rate estimator reports a delta of -0.0588. One zero-risk Defensive
episode contributes a paired delta of -0.95 and reverses the result.

## Decision

Use a paired cluster bootstrap of pooled risk-conditioned rates.

Each frozen `(opponent, pair_index, learner_side)` case remains one resampling
cluster. For the observed estimate and every bootstrap replicate:

1. sample paired case clusters with replacement;
2. sum protective-presence decisions and risk decisions separately for each
   policy;
3. calculate each policy's pooled protective-presence rate;
4. subtract the Strong Base rate from the Defensive rate.

Zero-risk cases contribute zero opportunities rather than an invented 0%
success rate. A replicate is invalid only if an entire sampled policy has zero
risk decisions. The evaluator must fail closed if no valid estimate or interval
can be formed.

The gate remains unchanged: the lower bound of the 95% paired bootstrap
interval must be strictly greater than zero.

## Alternatives Rejected

### Exclude zero-opportunity episodes and average episode rates

This removes the most obvious error but still gives an episode with one risk
decision the same weight as an episode with hundreds. It remains noisy and
does not estimate the documented aggregate risk-conditioned rate.

### Keep the estimator and train against it

This would optimize the model around a measurement artifact. It would also
penalize policies that prevent or shorten risk states, which conflicts with the
Defensive product behavior.

## Scope

The change is limited to the protective-presence point estimate and confidence
interval in `botcolosseo.evaluation.defensive`.

It does not change:

- the policy, training data, distillation, interpolation grid, or checkpoints;
- the frozen validation cases or pair identities;
- any skill-retention, denial/recovery, camping, objective, integrity, or
  test-isolation gate;
- the requirement that the confidence-interval lower bound be positive;
- the legal public-observation boundary.

The summary will identify the estimator as
`paired_cluster_bootstrap_pooled_ratio_v1` so downstream evidence is explicit.
The selection and evidence-audit paths will require that identity.

## Recalculation and Evidence

Existing completed episode ledgers are reusable. Re-running the evaluation CLI
must load the hash-bound records, recompute the summary and manifest, and run
selection without replaying episodes.

A read-only 10,000-sample calculation on the current smoke ledgers gives:

| Alpha | Pooled delta | 95% paired cluster bootstrap CI |
| --- | ---: | ---: |
| 0.25 | +0.1396 | [-0.0130, +0.2132] |
| 0.50 | -0.0655 | [-0.3649, +0.0397] |
| 0.75 | +0.0139 | [-0.2285, +0.1198] |

Therefore this correction does not make the current candidate pass. Alpha 0.25
has a positive effect estimate, but the 20-episode smoke evidence is not yet
strong enough for the unchanged confidence gate. The failed result must remain
visible, and any subsequent training or gate-design decision requires separate
evidence and review.

## Tests

Unit tests must prove:

- the pooled point estimate matches hand-calculated counts;
- a zero-risk case is not treated as a 0% failure;
- resampling is deterministic for a fixed seed;
- incomplete or entirely opportunity-free paired data fails closed;
- the existing positive, camping, missing-event, duplicate, and protocol-error
  cases retain their expected outcomes.

The existing real smoke ledgers will then be recomputed as an integration
check. No new long experiment is authorized by this design.
