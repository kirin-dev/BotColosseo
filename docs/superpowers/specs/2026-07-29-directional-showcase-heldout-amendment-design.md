# Directional Showcase Heldout Amendment

## Status and timing

Approved amendment to
`2026-07-29-directional-showcase-admission-design.md` for the product-only
Aggressive Showcase admission.

This rule was chosen after inspecting the first frozen Aggressive heldout
report. It is therefore a post-heldout product decision, not a preregistered
research threshold. The manifest and public claims must state that boundary.

## Frozen evidence

The 600k Aggressive heldout evaluation completed without protocol errors:

- report:
  `runs/extraction/styles/aggressive/evaluation-v2/`
  `candidate-0600000-heldout.json`;
- report SHA-256:
  `f9a05f916c9013ab3117bbd70559d07500a569fc41984a55a4b2e2ba1f6d9976`;
- Strong extraction rate: `0.8583`;
- Aggressive extraction rate: `0.8167`;
- extraction-rate delta: `-0.0417`, which passes the unchanged `-0.10`
  retention limit;
- protocol inconsistencies: `0`.

The original style heldout gate fails only
`heldout_worst_opponent_retention = -0.1333`. That value does not represent a
regression against the failing opponent. Strong wins `0.2333` against
Explorer, while Aggressive wins `0.2667`. The failure comes from the existing
formula's absolute `0.40` floor, which is higher than the frozen Strong Base's
own capability on that case family.

## Approved product capability rule

The research heldout gate and its `0.40` floor remain byte-for-byte unchanged.
Research selection, release freezing, and official-test paths continue to use
it.

Directional Showcase admission uses a separate relative-retention check:

```text
for each opponent:
    styled_win_rate >= max(0, strong_win_rate - 0.20)
showcase_worst_opponent_relative_margin =
    min(styled_win_rate - max(0, strong_win_rate - 0.20))
```

The check passes only when the minimum margin is non-negative. It asks whether
the style preserves the frozen Strong policy's observed per-opponent
capability within 20 percentage points; it does not invent an absolute
capability level that Strong itself failed to reach.

For the frozen 600k evidence, the minimum relative margin is `+0.0667`, against
Defensive. The Explorer comparison improves by `+0.0333`.

The remaining Showcase heldout checks are unchanged:

- extraction-rate delta must be at least `-0.10`;
- protocol inconsistencies, Actor privilege violations, and test-case access
  must remain zero;
- paired cases, policy checkpoints, Strong Base, scenario, protocol, and
  report hashes must match.

## Manifest truthfulness

`showcase-admission.json` must record:

- `admission_kind: directional_showcase`;
- `admission_rule_timing: post_heldout_product_review`;
- `showcase_eligible: true`;
- `research_gate_passed: false`;
- `research_validation_failed_checks: [style_ci_lower]`;
- `original_heldout_gate_passed: false`;
- `original_heldout_failed_checks:
  [heldout_worst_opponent_retention]`;
- both the unchanged research heldout checks and the separate Showcase
  heldout checks;
- per-opponent Strong and Aggressive win rates and relative margins;
- all checkpoint and evidence hashes.

The downstream prerequisite validator must require these exact fields and
hashes. A legacy or partially written admission fails closed.

## Unchanged boundaries

- No evaluation report, validation case, heldout case, checkpoint, reward,
  policy, or scene element is modified.
- No new training or official-test access is authorized.
- `selection.json` remains absent because the candidate is not
  research-eligible.
- Official/research release code remains unchanged and cannot consume
  `showcase-admission.json`.
- Defensive and Explorer continue to initialize independently from the frozen
  Strong Base. The admission changes scheduling only.

## Public claim

The Showcase may say that Aggressive preserves heldout extraction capability
and per-opponent capability within the product admission tolerance while
showing positive directional combat-chain evidence.

It must also say that:

- the predefined validation confidence interval overlaps zero;
- the original research heldout floor failed;
- the relative heldout rule was adopted after heldout review for product
  demonstration only.

It may not claim statistical significance, a passed research gate, or an
official-test result.

## Verification

- Recompute and preserve both the original research heldout result and the
  Showcase relative-retention result from the same frozen reports.
- Confirm the original validation and heldout report hashes are unchanged.
- Confirm `showcase.pt` hashes to the original 600k candidate.
- Confirm the downstream style runners accept the completed admission.
- Confirm research selection and release freezing still reject it.
- Run unit tests, Ruff, shell syntax checks, and the full existing test suite.
