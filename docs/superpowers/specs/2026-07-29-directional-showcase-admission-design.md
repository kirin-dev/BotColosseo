# Directional Showcase Admission Design

## Status

Approved product-first fallback for the
`feat/crystal-run-extraction-v3` branch. It does not change game mechanics,
training rewards, validation cases, research thresholds, or official-test
discipline.

## Evidence

The original 600k Aggressive policy is the strongest style candidate:

- ten of eleven frozen validation checks pass;
- paired task retention is 0.9346;
- the paired style difference is +0.1009;
- the complete combat-chain delta is +0.0417;
- valid-hit-per-attack improves rather than regresses;
- 50 paired cases improve, 35 regress, and 155 are unchanged;
- it creates 19 complete kill-to-cache-to-extraction chains that Strong misses
  while losing 9 that Strong completes;
- the sole failed research check is `style_ci_lower = -0.0090 > 0`.

The approved 1.5x reward calibration did not repair that confidence interval:
its 100k and 200k style differences fell to +0.0689 and +0.0488. Calibration
therefore stops without further scaling, retraining, or threshold changes.

## Two independent decisions

Research selection remains unchanged. The 600k candidate is not statistically
eligible, no passing `selection.json` is fabricated, and no official-test or
research-release command may accept it.

A separate Showcase admission may approve the learned policy for product
demonstration and for unblocking the remaining style-development pipeline. Its
manifest uses `admission_kind: directional_showcase`, records
`research_gate_passed: false`, and binds the unchanged candidate and evidence
hashes.

## Fail-closed Showcase gate

Aggressive Showcase admission requires all of the following:

1. the frozen validation gate fails only `style_ci_lower`;
2. paired task retention, extraction delta, mean-value ratio, worst-opponent
   retention, positive style mean, CI upper, both anti-hacking checks, and
   protocol integrity pass unchanged;
3. positive paired cases outnumber negative paired cases;
4. newly created complete kill-to-cache-to-extraction chains outnumber lost
   chains;
5. the existing paired heldout capability gate passes;
6. Actor privilege violations and test-case access remain zero;
7. checkpoint, Strong Base, scenario, protocol, validation, and heldout hashes
   match.

The admission CLI copies the exact learned candidate to `showcase.pt` and
writes `showcase-admission.json`. It refuses to overwrite drifted artifacts.

## Downstream and release boundary

Defensive and Explorer training may start when either:

- Aggressive has a normal eligible research `selection.json`; or
- Aggressive has a valid directional `showcase-admission.json`.

This prerequisite changes scheduling only. Defensive and Explorer still
derive independently from the same frozen Strong Base and use their unchanged
training rewards and evaluation gates.

Showcase rendering may resolve `showcase.pt` through the admission manifest.
Research selection, release freezing, official-test sealing/running, and any
claim of statistical significance continue to require normal eligible
selection manifests and must reject Showcase admission.

## Public claims

The public demonstration may say that the learned Aggressive adapter shows a
directional tendency and provide the paired mean, chain counts, capability
retention, and representative replay. It must also state that the predefined
95% confidence lower bound overlaps zero.

It may not say that Aggressive passed the full research gate, achieved
statistically significant style separation, or supports an official-test
claim.

## Success criteria

- The original validation and calibration reports remain byte-unchanged.
- `showcase.pt` hashes to the admitted 600k candidate.
- The admission records the sole research failure explicitly.
- Heldout capability and all non-CI validation checks pass.
- Downstream runners accept the admission while official/research release
  paths reject it.
- No runtime governor, scripted override, privileged input, or scene change is
  introduced.
