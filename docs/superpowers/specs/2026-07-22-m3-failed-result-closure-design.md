# M3 Failed-Result Closure Design

**Date:** 2026-07-22
**Status:** Approved direction (Approach A)
**Scope:** Package the completed M3 official evaluation honestly when a frozen
capability gate fails. This design does not change training, selection,
thresholds, raw rows, or the official PASS/FAIL decision.

## Problem

The M3 official evaluator completed all 1,340 scheduled episodes and wrote a
hash-bound identity, episode ledger, summary, and manifest. The evidence is
complete and protocol-clean, but the frozen
`historical_worst_case_improved` capability gate failed. Returning exit code 1
is correct.

The serial pipeline currently runs under `set -e`. The evaluator's correct
nonzero capability result therefore stops the driver before independent
integrity auditing, figure rendering, and the final result marker. The current
audit API also requires capability PASS, so callers cannot distinguish
"cryptographically and structurally valid FAIL evidence" from damaged or
incomplete evidence.

## Goals

1. Preserve `summary.passed == false` and the pipeline's final nonzero status.
2. Verify a completed FAIL result against raw rows, hashes, repository bindings,
   schedule completeness, and protocol integrity.
3. Render the existing cross-play and pool-history evidence for either an
   integrity-clean PASS or integrity-clean FAIL result.
4. Publish a concise M3 milestone record that states the single failed gate and
   the supporting numbers without claiming Strong Base success.
5. Reuse the completed 1,340-row ledger; no evaluation or training rerun is
   required.

## Non-goals

- Changing any frozen M3 threshold or gate definition.
- Selecting a checkpoint after inspecting M3 test results.
- Treating a capability FAIL as M3 completion or enabling M4 to claim a passed
  Strong Base.
- Creating a second evaluator, audit algorithm, or permissive evidence format.
- Rendering hand-edited numbers that are not derived from committed evidence.

## Design

### Audit result model

`audit_m3_evidence` will accept a keyword-only
`require_capability_pass: bool = True` argument. Its existing default remains
strict and backward-compatible. The audit CLI will expose the same choice as
`--integrity-only`; omitting the flag retains the current strict behavior.

Every invocation performs the same integrity work:

- validate identity, summary, manifest, and episode-ledger presence;
- verify manifest hashes and selected checkpoint/pool/baseline bindings;
- load all raw rows and recompute the complete M3 summary;
- require exact equality between recomputed and stored summaries;
- require an official, complete schedule with no protocol or artifact
  inconsistencies;
- when an artifact root is supplied, verify every repository binding.

When `require_capability_pass` is true, a recomputed capability FAIL continues
to raise `ValueError`. When false, an integrity-clean capability FAIL returns a
structured result with `integrity_passed: true`, `capability_passed: false`,
`passed: false`, and the exact failed gate names. Integrity failures always
raise; there is no mode that renders corrupted evidence.

### Rendering

`render_m3_evidence.py` will call the audit in integrity mode. It may render the
existing hash-derived cross-play heatmap, PFSP pool history, and canonical
matrix for either capability outcome. Its JSON output includes the audit
status, so downstream automation cannot mistake rendered FAIL evidence for a
PASS.

The figures describe the completed experiment and do not add a green success
badge. The milestone Markdown record will be written from verified summary
values and will explicitly label M3 as `FAIL`.

### Pipeline control flow

The driver will temporarily disable immediate exit only around the official
evaluator, capture `EVALUATION_STATUS`, and restore `set -e`. It will then:

1. run the integrity audit in capability-optional mode;
2. render the evidence bundle;
3. print `M3 PIPELINE PASS` and return zero only when the evaluator returned
   zero;
4. otherwise print `M3 PIPELINE COMPLETE: CAPABILITY FAIL` and return the
   evaluator's nonzero status.

An evaluator crash that fails to produce complete, auditable artifacts will
still stop at the integrity audit. Thus the completion marker distinguishes a
valid experimental FAIL from a runtime failure without weakening either.

### Public documentation and artifacts

The tracked result closure will include:

- `docs/milestones/m3.md` with the frozen gate table and limitations;
- `reports/m3/official/{run-identity,summary,manifest}.json`;
- the 960-KiB raw `episodes.jsonl` ledger so the committed summary remains
  independently recomputable;
- `reports/m3/strong-base-selection.json`, final cross-play evidence, and pool
  history needed to reproduce the figures;
- `docs/assets/m3-crossplay-heatmap.png` and
  `docs/assets/m3-pfsp-pool-history.png`;
- README and Plan status text stating engineering completion and capability
  failure separately.

Generated validation matrices and checkpoints remain untracked.

## Error handling

- Missing, malformed, hash-mismatched, dirty, incomplete, or protocol-invalid
  evidence is an integrity error and prevents rendering.
- A complete official result with one or more false capability gates is a valid
  audited FAIL and remains nonzero at the pipeline boundary.
- Existing artifacts are never deleted or rewritten to make the result pass.
- Re-running the closure commands is deterministic and does not open the
  environment or consume additional test episodes.

## Testing

Unit tests will prove:

1. strict audit still rejects a recomputed capability FAIL;
2. integrity mode accepts the same hash-consistent FAIL and returns its failed
   gates;
3. integrity mode still rejects tampering, incompleteness, and protocol errors;
4. the renderer requests integrity mode and exposes both integrity and
   capability status;
5. the shell pipeline captures evaluator status, audits and renders before its
   final conditional exit, and never emits the PASS marker on failure;
6. public documentation cannot claim that M3 passed.

Fresh unit, Ruff, shell-syntax, real-artifact audit, deterministic render, and
repository-hygiene checks form the completion gate. No long experiment is part
of this change.
