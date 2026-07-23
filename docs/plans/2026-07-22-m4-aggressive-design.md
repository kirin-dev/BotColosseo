# M4 Aggressive vertical-slice design

## Goal

Produce a visibly more aggressive bot without allowing style reward to replace
the Crystal Run objective. The M3 `policy-0200000` checkpoint is used honestly
as an integrity-qualified style-base candidate; it is not relabeled as a
passed Strong Base.

## Policy shaping

- Freeze the complete fair-observation CNN, scalar encoder, GRU, and base policy.
- Add a zero-initialized bottleneck residual adapter and a copied policy head.
- Train the adapter, copied head, and critic only.
- Penalize `D_KL(style || base)` on valid recurrent tokens.
- Preserve the original task reward and add capped Aggressive shaping.

The initial styled policy is exactly equal to the base policy. The adapter is
preferred over GRU fine-tuning because it is cheap, auditable, and limits
catastrophic forgetting. GRU unfreezing remains a fallback only if the style
signal is too weak after evaluation.

## Aggressive reward v1

Training uses only the learner's public action, public `has_core` observation,
and public gameplay events. A valid hit rewards effective engagement; the first
valid hit after a cooldown also rewards engagement initiation. Forward attacks
are rewarded only when they produce a valid hit. Attacks without a valid hit
are penalized, with an additional penalty while carrying the core. Every term
has a per-episode cap and is logged separately.

Distance and visibility remain evaluation-only signals until they can be
derived without adding privileged inputs to the policy/reward path.

## Evidence boundary

The first 400k run is a production candidate, not an M4 pass. Promotion still
requires paired Base/Aggressive validation, skill-retention checks, style
metrics, anti-hacking review, and real comparison media. Historical M2/M3 gate
failures remain declared technical debt.
