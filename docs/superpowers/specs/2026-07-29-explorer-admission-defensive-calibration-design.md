# Explorer Admission and Defensive Calibration Design

## Status

Approved product-first closeout for the remaining learned styles on
`feat/crystal-run-extraction-v3`.

This design does not change the ViZDoom scene, action space, task reward,
Strong Base, Actor observations, KL constraint, validation cases, research
thresholds, or official-test boundary.

## Frozen evidence

### Explorer

The 600k Explorer candidate is the validation Pareto winner:

- checkpoint:
  `runs/extraction/styles/explorer/candidate-0600000.pt`;
- checkpoint SHA-256:
  `a31e75b57bda68c60e9f745c929212587279c8bd71f73465d9711415ee520740`;
- validation report SHA-256:
  `b7d83d303debdb2227b66d20827e05b7e3f73b2b7a8c350d0870f46e5c25ba32`;
- paired task retention: `0.9439`;
- extraction-rate delta: `+0.0083`;
- paired Explorer difference: `+0.0500`;
- bootstrap interval: `[-0.0137, +0.1158]`;
- upgrade-to-extraction delta: `+0.0417`;
- positive, negative, unchanged paired cases: `33`, `27`, `180`;
- newly created versus lost upgrade-to-extraction chains: `23` versus `13`;
- protocol inconsistencies and test-case access: `0`.

Ten of eleven frozen validation checks pass. Only `style_ci_lower` fails.
Explorer heldout evidence has not been accessed when this design is frozen.

### Defensive

The original Defensive configuration does not produce the intended direction:

- 200k paired style difference: `-0.0614`;
- 400k paired style difference: `-0.0734`;
- successful disengagement rate: Strong `0.6585`, 200k `0.6174`, 400k
  `0.6000`;
- task retention, extraction, anti-inactivity, timeout, and protocol checks
  still pass.

Continuing the unchanged configuration to 600k is rejected. The failure is
not a task-capability collapse; the style reward overvalues the already-easy
meaningful extraction event relative to the distinguishing disengagement
behavior.

## Explorer directional admission

The existing admission implementation becomes style-aware without weakening
research selection.

Explorer Showcase admission requires:

1. frozen validation fails only `style_ci_lower`;
2. every capability, positive-mean, CI-upper, anti-hacking, and protocol check
   passes unchanged;
3. positive paired Explorer scores outnumber negative scores;
4. newly created upgrade-to-extraction chains outnumber lost chains;
5. the product heldout gate, frozen before heldout access, passes:
   - extraction-rate delta is at least `-0.10`;
   - for every opponent, styled win rate is at least
     `max(0, Strong win rate - 0.20)`;
   - protocol errors are zero;
6. Actor privilege violations and test-case access are zero;
7. checkpoint, Strong Base, scenario, protocol, validation, and heldout hashes
   match.

The unchanged research heldout gate is also computed and recorded but does
not determine product admission. Its result is not known when this rule is
frozen.

The manifest records:

- `admission_kind: directional_showcase`;
- `admission_rule_timing: pre_heldout_product_rule`;
- `policy: explorer`;
- `showcase_eligible: true`;
- `research_gate_passed: false`;
- the original research validation and heldout checks;
- the separate Showcase heldout checks;
- paired direction and upgrade-chain counts;
- all artifact hashes and observation-integrity fields.

The exact 600k checkpoint is copied to
`runs/extraction/styles/explorer/showcase.pt`. No `selection.json` is
fabricated.

## Defensive targeted calibration

### Reward change

Create an isolated calibration configuration with:

```yaml
style_reward_overrides:
  defensive:
    risk_disengagement: 0.30
    combat_with_value: -0.030
```

The defaults remain:

- `meaningful_extraction: 0.20`;
- `empty_idle: -0.003`;
- all event caps unchanged;
- `style_reward_scale: 1.0`.

Only the two distinguishing Defensive coefficients change. Task reward,
optimizer, architecture, KL and residual penalties, opponent schedule,
training cases, and observation boundary remain unchanged.

The training CLI accepts optional per-style reward dataclass overrides,
rejects unknown or non-finite values, and writes the fully resolved reward
configuration into `summary.json`.

### Isolated run

- Output:
  `runs/extraction/styles/defensive-calibration-v2`;
- initialization: a fresh residual adapter over the frozen selected Strong
  Base, not either negative-direction Defensive adapter;
- configured maximum: 400k;
- first stop: 200k;
- checkpoint interval: 200k;
- no test or heldout access during training.

The runner resumes only its own `latest.pt` and refuses mixed lineage.

### Stop and promotion rules

At 200k, evaluate the unchanged 240-case validation protocol.

- If the strict research gate passes, run paired heldout selection normally.
- If every non-CI check passes, paired style mean is positive, positive paired
  cases outnumber negative cases, and newly created complete Defensive
  Showcase chains outnumber lost chains, evaluate heldout for directional
  admission.
- A complete Defensive Showcase chain means the same episode contains at
  least one successful low-resource disengagement and a meaningful
  extraction. Extraction terminates the episode, so the disengagement
  necessarily precedes it.
- If paired style mean is non-positive, stop the calibration permanently.
- Resume once to 400k only when the paired mean is positive and all
  capability, anti-hacking, and protocol checks pass, but the validation
  directional evidence criteria are not yet all positive.
- Failure at 400k ends calibration. Reward weights and thresholds are not
  changed again.

Directional Defensive admission uses the same pre-heldout relative capability
rule as Explorer. A successful calibrated candidate is copied into the
canonical Defensive Showcase location with its calibration lineage and hashes
preserved.

## Code boundaries

- Add style-specific paired direction and Showcase-chain evidence helpers.
- Generalize the admission CLI to `aggressive`, `defensive`, and `explorer`
  while preserving Aggressive's post-heldout amendment exactly.
- Add a fail-closed style Showcase artifact resolver for rendering.
- Add optional reward overrides and an isolated Defensive calibration runner.
- Leave `style_validation_gate`, `style_heldout_gate`,
  `freeze_extraction_release`, official-test code, and scenario assets
  unchanged.

## Public claims

Explorer may be described as a learned directional style if admission passes:
it improves upgrade-to-extraction behavior while retaining task capability.
The public text must state that the predefined 95% confidence lower bound
overlaps zero.

Defensive may be described only from the final calibrated evidence. The
original 200k/400k negative-direction runs remain preserved as failed
development evidence and are not used in Showcase media.

No directional admission may be described as statistically significant,
research-gate eligible, or official-test evidence.

## Verification

- Unit-test style-specific direction counts, chain counts, report identity,
  reward override validation, and fail-closed manifest resolution.
- Verify the Explorer rule is committed before running Explorer heldout.
- Verify existing Aggressive admission remains valid byte-for-byte.
- Verify research release freezing rejects every directional admission.
- Run targeted tests, Ruff, shell syntax checks, and the full test suite.
- Run staged experiments under the current duration policy and preserve all
  validation/heldout reports and hashes.
