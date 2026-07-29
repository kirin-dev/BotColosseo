# Extraction Style Ablation Design

**Date:** 2026-07-29
**Status:** approved by standing user authorization

## Question

For a frozen capable Strong Actor, which regularizers are needed to obtain a
style shift without losing task capability?

The primary style objective is:

```text
task reward + style reward
  - beta_kl * KL(style || Strong)
  - rho_residual * residual magnitude
```

This ablation tests the two regularizers rather than adding unrelated model
architectures.

## Matrix

Each of Aggressive, Defensive, and Explorer is evaluated under:

| Variant | `beta_kl` | `rho_residual` | Interpretation |
|---|---:|---:|---|
| Full | 0.08 | 0.01 | primary method |
| Reward + KL | 0.08 | 0.00 | remove residual-magnitude control |
| Reward only | 0.00 | 0.00 | remove both retention regularizers |

The existing Full 200k checkpoints and validation reports are reused. Six new
training runs are required.

## Controlled comparison

All variants preserve:

- the exact selected Strong checkpoint and SHA-256;
- the fair Actor observation boundary;
- seed, training cases, opponent schedule, PPO hyperparameters, adapter
  architecture, style reward, and action space;
- the original learning-rate schedule horizon: 600k for Aggressive/Explorer
  and 400k for the calibrated Defensive configuration;
- a common comparison checkpoint at 200k environment steps.

Training stops at 200k without changing the configured schedule horizon. This
makes each ablation directly comparable to the already saved Full 200k
checkpoint instead of silently accelerating its learning-rate decay.

Defensive uses the published calibrated reward coefficients:
`risk_disengagement=0.30` and `combat_with_value=-0.030`. Aggressive and
Explorer use their published reward configurations unchanged.

## Evaluation

Every one of the nine cells uses the same frozen 240-case validation protocol
and paired Strong report. Report:

- paired style shift and its 95% bootstrap interval;
- paired task retention;
- extraction-rate delta;
- mean extracted-value ratio;
- worst-opponent retention margin;
- style-specific reward-hacking checks;
- protocol, observation, and test-access integrity.

The primary public table shows paired style shift and task retention. A
machine-readable report retains all gate values and disclosed failures.
Style-shift scales are style-specific and are not compared across styles.

No official-test cases are accessed. Heldout evaluation is not required for
this coefficient ablation because the experiment asks a matched validation
mechanism question, not candidate selection or release promotion.

## Execution

Use two independent GPU lanes. Each lane runs three train-then-evaluate jobs
sequentially:

- GPU 0: Reward-only Aggressive; Reward-only Explorer; Reward+KL Defensive.
- GPU 1: Reward-only Defensive; Reward+KL Aggressive; Reward+KL Explorer.

Each job is resumable from its own `latest.pt`, refuses configuration drift,
and writes isolated logs and artifacts under
`runs/extraction/ablations/<variant>/<style>/`.

Expected wall time is about 4.5 hours. The detached pipeline is checked at 50%
of estimated wall time, about 2 hours 15 minutes after launch. A later check is
made at the revised halfway point of the remaining estimate if needed.

## Publication

After all six validation reports pass integrity checks:

1. generate `reports/extraction/style-ablation.json`;
2. add a compact ablation table to the GitHub Pages Results section;
3. add the same table to the bilingual README;
4. disclose failures without changing thresholds or selecting a favorable
   subset;
5. rerun documentation, unit, artifact, and online-page verification.

## Completion

The ablation task is complete only when:

- all six new 200k checkpoints and validation reports exist;
- all nine cells bind the same Strong, protocol, scenario, and case identities;
- no cell accessed test cases or violated the Actor observation boundary;
- the machine-readable summary and public tables agree exactly;
- the updated page is committed, pushed, deployed, and verified online.
