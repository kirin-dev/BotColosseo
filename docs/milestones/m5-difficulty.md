# M5 Difficulty controller: all-style product matrix passed

The deterministic public-observation controller and hybrid composition passed
the complete frozen product matrix on 2026-07-23.

![All-style difficulty performance](../assets/showcase/m5-hybrid-all-style-difficulty.png)

| Policy | Easy | Normal | Hard (native) |
|---|---:|---:|---:|
| Strong Base performance | 0.820 | 0.878 | 0.955 |
| Aggressive performance | 0.830 | 0.890 | 0.955 |
| Defensive performance | 0.818 | 0.878 | 0.938 |
| Explorer performance | 0.830 | 0.880 | 0.960 |

The audit projects five hash-bound source ledgers into exactly 1,200 unique
validation episodes: four policies, three tiers, and 100 cases per cell. It
reuses the original 600 Strong Base/Aggressive rows and the two 100-row hybrid
Hard cells, then adds only the 400 missing Defensive/Explorer Easy/Normal
episodes. Every source identity, case, seed, side, scenario, and protocol field
is checked before aggregation.

All policies passed aggregate monotonicity and the four-of-five per-opponent
monotonicity rule. Objective capability, style mechanism coverage, and
same-tier hybrid Skill Retention also passed. The minimum retention across
every hybrid policy, tier, and opponent was 92.3%. There were zero protocol
inconsistencies, and `test_cases_accessed` is false.

Easy adds a two-decision reaction delay and slower policy updates; Normal adds
one decision of reaction delay. Hard is the native learned or hybrid policy.
The controller changes no health, damage, observation, checkpoint weight, or
hidden game state. Defensive and Explorer retain their explicit
`hybrid governor` labels.

The earlier 1,800-row learned-only design is retained as historical context.
Its exact cross-run Strong Base outcome assumption is contradicted by current
artifacts and is not the product acceptance route.

The compact [combined audit](../../reports/m5/difficulty/hybrid-all-style-summary.json),
per-style run/summary/manifests, checkpoint and configuration hashes are
tracked in Git. Large per-decision ledgers are distributed separately through
the hash-bound
[raw-evidence release record](../../reports/m6/hybrid-difficulty-evidence-release.json).
