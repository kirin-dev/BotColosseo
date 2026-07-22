# M4 Aggressive Success-Filtered Distillation Design

## Decision

The first reward-only Aggressive candidate preserved most task capability but did not learn a visible attack style. The approved replacement is a two-stage route: success-filtered teacher distillation into the residual style path, followed by bounded PPO continuation.

This is a style-acquisition change, not a new base-policy architecture. The M3 actor remains the frozen capability anchor and the existing M4 paired evaluator remains the acceptance authority.

## Evidence and cause

- The M2 behavioral-cloning demonstrations contain no learner attack actions.
- The first 400k-step Aggressive candidate completed objectives but produced zero measured attacks in the M4 smoke evaluation.
- `AggressiveDuelTeacher` can propose attacks from privileged state, but some proposals can be occluded or otherwise invalid.

Reward shaping alone therefore has too little useful on-policy attack signal. Copying every privileged teacher attack would instead teach invalid or wall-facing attacks.

## Data generation

Generate training-only trajectories with `AggressiveDuelTeacher` controlling the learner against the existing duel opponent set. Each transition records the pre-action public observation, recurrent boundary, teacher action, and post-step learner events.

Training labels follow these rules:

1. Keep an attack label only when that action produces a learner `VALID_HIT` event.
2. Exclude unsuccessful teacher attacks from the supervised loss.
3. Keep a bounded sample of teacher non-attack actions as negative/context examples so the policy does not collapse to always attacking.
4. Preserve episode order and boundaries so the frozen recurrent base receives valid history.
5. Bind outputs to source manifest, scenario, configuration, seed, and `test_cases_accessed=false` metadata.

No validation or test cases may enter adapter training.

## Adapter pretraining

Load the selected M3 base checkpoint into `StyledActorCritic`. Freeze the full base actor and optimize only the residual adapter and copied style policy head. The supervised objective is masked cross-entropy on successful attack positives plus sampled non-attack context, with a KL term to the frozen base distribution to retain general behavior.

The output checkpoint must remain loadable by the existing styled PPO and evaluation paths. A warm-start argument will let `train_league --style aggressive` load these style weights while still verifying the M3 base identity.

## PPO continuation

Continue training in the original duel task with the existing task reward, Aggressive shaping, KL regularization, and anti-hacking penalties. The distillation checkpoint supplies attack competence; PPO is responsible for integrating it with objective completion and opponent diversity.

The continuation is bounded and checkpointed. It does not modify the frozen public-observation base actor.

## Acceptance criteria

Before a long run:

- Generated data contains successful attack positives and no unsuccessful attack labels.
- Data metadata proves training-only access.
- Pretraining changes only adapter/style-head parameters.
- A small smoke run raises attack probability on retained positives without collapsing representative negatives.
- The warm-started PPO path resumes and writes a valid styled checkpoint.

Final acceptance uses the existing paired M4 evaluation: protocol integrity, skill retention, per-opponent retention, engagement shift, valid attack rate, and controlled objective chase. A failed candidate is preserved as evidence but is not promoted to the showcase.

## Scope boundary

This iteration implements Aggressive only. Defensive and Explorer reuse the proven residual-style infrastructure after Aggressive passes. No inference-time action override, privileged runtime observation, or test-set optimization is permitted.

## Approved Pareto interpolation addendum

Validation showed that the distilled endpoint has visible but excessive
aggression, the 200k PPO endpoint retains capability but has sparse aggression,
and the 400k endpoint removes greedy attack behavior. The approved correction
is a fixed checkpoint interpolation between the distilled and 200k endpoints.

For `alpha` in `{0.25, 0.50, 0.75}`, interpolate only `adapter.*` and
`policy.*` tensors as `(1 - alpha) * ppo_200k + alpha * distilled`. Copy the
distilled checkpoint's frozen base tensors unchanged and verify them against
the M3 base identity. Each output records alpha, both source checkpoint hashes,
the M3 base hash, and scenario hash. It remains a deterministic checkpoint,
not a runtime action override.

Run the existing 20-episode validation smoke for each alpha. Promote the
candidate that passes all gates; if none passes, rank candidates by number of
passed gates, then Skill Retention, then engagement shift. A full 200-episode
evaluation is permitted only after a smoke candidate passes every gate.
