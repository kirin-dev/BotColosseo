# M5 Defensive Data-Gate Waiver Design

**Status:** route approved; written specification awaiting final review  
**Date:** 2026-07-23  
**Scope:** one immutable waiver for the current Defensive train manifest

## 1. Decision

The project owner approved proceeding from the completed 50,000-transition
Defensive dataset even though it recorded 95 verified denial/recovery windows
against the predefined minimum of 100. This is a five-window shortfall in a
training-data admission check, not a claim that the check passed.

The authorized manifest is exactly:

```text
data/generated/m5/defensive/train-manifest.json
sha256: 6d40022516ed16ace94690dcac10d38044d5abbbef3921baf4e787346c0775b3
```

It contains 50,000 transitions, 4,854 selected risk transitions, 2,427
selected no-risk transitions, 145 successful windows, 95 denial/recovery
windows, all five opponents, both learner sides, and
`test_cases_accessed: false`. Every data gate except
`denial_recovery_windows` passed.

## 2. Evidence semantics

The source manifest remains byte-identical and retains `passed: false`. A
separate committed waiver records the manifest hash, observed and required
window counts, the single failed gate, the owner's approval, and the unchanged
scenario/Base identities. Reports must distinguish:

- `data_gate_passed: false`;
- `data_waiver_applied: true`;
- downstream offline, smoke, and formal gate outcomes.

No README, milestone record, audit, or model card may describe the original
100-window data gate as passed.

## 3. Admission contract

Defensive distillation continues to reject a failed manifest by default. It
accepts `--data-waiver` only when all of these conditions hold:

1. waiver schema and stage are exact;
2. waiver manifest SHA-256 matches the loaded manifest;
3. the manifest has exactly 50,000 transitions and exactly 95 observed versus
   100 required denial/recovery windows;
4. `denial_recovery_windows` is the only false data gate;
5. risk-transition, balance, opponent, side, and completeness gates pass;
6. scenario and Strong Base hashes match the waiver and checkpoint;
7. neither artifact accessed test cases.

The waiver is single-use by identity: a regenerated or edited manifest cannot
inherit it. There is no generic `--ignore-gates` option.

## 4. Downstream gates

The waiver changes no model-selection or publication threshold. The existing
requirements remain authoritative:

- Defensive offline agreement shift and no-risk control;
- fixed interpolation grid `0.25/0.50/0.75`;
- all-gate 20-episode smoke eligibility;
- unchanged 200-episode formal retention, paired style, denial/recovery,
  anti-camping, objective, protocol, and no-test-access gates.

If no alpha passes smoke, or the selected checkpoint fails formal evaluation,
the result remains failed and is not published as Defensive.

## 5. Artifacts and audit

Add one tracked waiver JSON under `reports/m5/defensive/`. Bind its hash into
the distillation summary. Extend the final M5 audit to require either a passing
data manifest or this exact valid waiver, while reporting the original failed
gate separately. The production script supplies the waiver explicitly; other
training callers retain strict behavior.

## 6. Verification

Tests must prove that distillation:

- accepts the exact approved manifest/waiver pair;
- rejects an absent waiver, changed manifest hash, changed counts, additional
  failed gate, scenario/Base mismatch, or test access;
- records the waiver hash without changing source artifacts;
- leaves all downstream thresholds and the fixed alpha grid unchanged.

The implementation uses native Codex planning, testing, verification, and Git
workflows. No other Superpowers skill is invoked.
