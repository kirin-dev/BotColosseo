# M5 All-Style Difficulty Extension

**Status:** Approved through delegated recommendation authority on 2026-07-23.

## 1. Gap

The first 600-episode difficulty run covers Strong Base and Aggressive. It
proves the controller mechanism but cannot by itself satisfy the M5
Style×Difficulty claim for Defensive and Explorer.

A single generic performance evaluator would be shorter to implement, but it
could not prove that protective presence and route diversity survive the
controller. The extension therefore keeps each style's native frozen metric.

## 2. Three evidence blocks

Use the same ten side-swapped validation pairs per opponent, three difficulty
profiles, scenario, decision limit, and retry policy.

1. Aggressive block: Strong Base versus Aggressive, 600 episodes, using the
   existing engagement evaluator and current formal ledger.
2. Defensive block: Strong Base versus the passing Defensive PPO checkpoint,
   600 episodes, using protective-presence attribution.
3. Explorer block: Strong Base versus the passing Explorer PPO checkpoint,
   600 episodes, using route attribution and normalized route entropy.

Total M5 Style×Difficulty evidence is 1,800 episodes. The existing Aggressive
block is reused by hash; it is not rerun merely to fit a new directory layout.
Defensive or Explorer blocks cannot start before their unchanged 200-episode
style gate passes.

## 3. Runtime

Each style evaluator wraps both Strong Base and the derived style checkpoint in
the same `DifficultyPolicy`. The wrapper receives only public Actor
observations. Privileged state remains confined to the existing offline metric
ledger.

Each block writes an independent resumable ledger keyed by:

```text
policy, difficulty, opponent, pair_index, learner_side
```

Identity binds style checkpoint, Strong Base, difficulty config, selected
cases, scenario, evaluator estimator, and zero test access.

## 4. Frozen acceptance

Every block must be complete and protocol-clean. For both Base and the style
policy:

- aggregate performance is Easy ≤ Normal ≤ Hard with the existing 0.03
  adjacent-tier tolerance;
- at least four of five opponents are monotonic;
- Easy objective rate is at least 70% of Hard;
- Normal objective rate is at least 85% of Hard.

At every difficulty:

- style Skill Retention is at least 0.85 overall and 0.75 per opponent;
- Aggressive engagement shift remains positive;
- Defensive protective-presence shift remains positive;
- Explorer route-entropy shift remains positive, retains all three credited
  routes, improves flank completion, and stays within the frozen efficiency
  bound.

The existing estimator and style-specific anti-hacking rules remain unchanged.
No difficulty can borrow a positive shift from another tier.

## 5. Combined audit

The combined audit requires:

- all three block manifests pass;
- Strong Base, style, scenario, config, case, estimator, ledger, and summary
  hashes match;
- exactly 1,800 unique episode identities;
- the Strong Base common outcome fields agree where the same deterministic
  difficulty case is represented by different style-specific ledgers;
- `test_cases_accessed: false` throughout.

The combined summary is the only artifact allowed to set
`difficulty_gate_passed: true` in M6 metrics.

## 6. Failure behavior

The current 600-episode Aggressive block remains immutable even if it fails.
No profile, tolerance, metric, or case changes after observing its result.
Likewise, a failed Defensive or Explorer block is retained and prevents the
M5/M6 all-style claim.

Implementation may be prepared while style training runs, but no long block is
launched without a passing upstream style checkpoint and a free GPU.
