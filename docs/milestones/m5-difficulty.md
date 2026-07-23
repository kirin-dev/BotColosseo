# M5 Difficulty controller: Base/Aggressive calibration passed

The deterministic public-observation controller passed its frozen
Strong Base/Aggressive validation block on 2026-07-23.

![Difficulty performance](../assets/showcase/m5-difficulty.png)

| Policy | Easy | Normal | Hard (native) |
|---|---:|---:|---:|
| Strong Base performance | 0.820 | 0.878 | 0.955 |
| Aggressive performance | 0.830 | 0.890 | 0.955 |
| Strong Base objective rate | 87% | 93% | 99% |
| Aggressive objective rate | 88% | 94% | 99% |

All 600 paired validation episodes completed with zero environment retries and
zero protocol inconsistencies. Both policies passed aggregate monotonicity,
objective capability, and the four-of-five per-opponent monotonicity rule.
Aggressive engagement remained above Strong Base at every tier:
`0.0497`, `0.0069`, and `0.0982` initiations per 100 decisions versus zero.

Easy adds a two-decision reaction delay and updates the policy every two
decisions. Normal adds one decision of reaction delay. Hard is the unchanged
native checkpoint. The controller changes no health, damage, observation,
checkpoint weight, or hidden game state.

This is a passing controller calibration for Strong Base and Aggressive, not
yet the complete M5 Style×Difficulty gate. Defensive and Explorer must first
pass their own style gates and then pass their native style metrics under all
three profiles. The frozen all-style protocol contains 1,800 episodes.

Raw ledgers, summaries, manifests, checkpoint hashes, configuration hash, and
zero-test-access evidence are tracked under
[`reports/m5/difficulty/`](../../reports/m5/difficulty/).
