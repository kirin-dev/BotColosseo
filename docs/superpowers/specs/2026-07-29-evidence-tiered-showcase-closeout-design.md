# Evidence-Tiered Showcase Closeout Design

## Status

Approved product-first closeout for `feat/crystal-run-extraction-v3`.

This design does not change the ViZDoom scene, game rules, action space,
Actor observations, task reward, style rewards, Strong Base, policy
architecture, validation cases, heldout cases, research thresholds, or
official-test boundary.

## Product objective

The immediate deliverable is a clean public GitHub Showcase containing one
capable Strong Bot and three visibly different learned styles:

- Aggressive converts combat into corpse-cache value and extraction;
- Defensive disengages under risk, preserves carried value, and extracts;
- Explorer searches broadly, upgrades its backpack, and converts useful
  exploration into banked value.

The Showcase must be easy to understand from the videos while preserving the
true evidence boundary. It is a product demonstration, not a replacement for
the frozen research release.

## Evidence tiers

Every published policy artifact has one explicit evidence tier.

### Research selection

`research_selection` means the unchanged validation and heldout research gates
passed. It may be used by the research release freezer and official-test
runner.

Strong remains in this tier. A style enters this tier only through the
existing strict selector.

### Directional Showcase admission

`directional_showcase` means the precommitted product admission rule passed,
but the full research gate did not. It may be used for validation-only media,
but never for release freezing or official test.

Aggressive retains its existing admitted artifact and evidence exactly.
Defensive may enter this tier only if its frozen calibration rule passes.

### Validation-demonstrated style

`validation_demonstration` is a new, lower product evidence tier. It means:

1. the checkpoint is a learned residual adapter over the exact selected
   Strong hash;
2. the 240-case validation report is complete, fair-observation-only,
   protocol-clean, and test-free;
3. capability, anti-hacking, and protocol checks pass on validation;
4. the paired style mean is positive;
5. the selected real replay contains the complete style-specific product
   chain;
6. every accessed heldout result and failed check is bound into the manifest;
7. the artifact is explicitly ineligible for research release and official
   test.

Explorer uses this tier. Its validation evidence shows a positive paired
Explorer shift, a positive majority of paired cases, and more newly created
than lost upgrade-to-extraction chains. Its heldout extraction-rate delta of
`-0.175` is published as a generalization limitation.

Explorer is not defined by extraction rate alone. Its product identity is the
combination of useful route coverage, real backpack upgrades, limited
unproductive wandering, and conversion of upgraded inventory into banked
value. Extraction remains a task-capability safeguard, not the sole style
metric.

Defensive uses this tier only if the final frozen 200k/400k validation
selection has a positive paired mean, passes capability and anti-hacking
checks, and produces a real disengagement-to-meaningful-extraction replay.
If those conditions fail, Defensive is not published as a learned style.

No lower tier may be called statistically significant, research-gate
eligible, heldout-robust, or official-test evidence.

## Contrastive replay selection

Showcase selection remains validation-only and uses real learned-policy
rollouts. It ranks cases by visible product meaning, not by generic task score
alone.

For each style, the selector pairs its episode with Strong on the same seed,
learner side, and scripted opponent. It first enforces the complete story,
then prefers a positive style-score delta over Strong.

- Aggressive requires valid hits, one kill, corpse-cache loot, and successful
  extraction in order.
- Defensive requires a successful low-resource disengagement followed by
  meaningful extraction; zero kills is preferred so the clip reads as
  preservation rather than generic combat.
- Explorer requires a genuine backpack upgrade followed by extraction; zero
  valid hits and zero kills are preferred so the clip reads as exploration
  rather than generic combat.
- Strong requires a representative high-value successful extraction.

The selection manifest records the paired Strong episode, style-score delta,
hard story checks, case identity, checkpoint hash, report hash, and evidence
tier.

## Replay reliability and fail-closed rendering

ViZDoom multiplayer replay can branch slightly across independent processes
even with the same scenario seed. Therefore a validation case identifies a
representative case distribution, not a guaranteed byte-identical movie.

Rendering follows these rules:

1. CUDA inference is warmed before the live multiplayer episode starts.
2. The online loop stores raw frames and immutable telemetry only.
3. Viewer overlays are drawn after the live episode ends.
4. A renderer may attempt the same preselected case at most five times.
5. Every attempt result is recorded.
6. The first attempt whose actual replay satisfies the policy's frozen story
   checks is accepted.
7. If all five attempts fail, no media artifact is published.

This retry is not candidate selection and does not access heldout or test. It
only makes the already selected validation case fail closed against the claims
shown in the final video.

Viewer overlays show both HP bars, fixed 20-point damage, ammunition, backpack
slots, carried and banked value, corpse-cache value, extraction progress,
current macro action, and persistent event banners. Defensive replays also
show when safe disengagement distance is created.

## Product manifest and audit

The final Showcase manifest binds:

- Strong plus all three style checkpoint hashes;
- each artifact's evidence tier and evidence manifest;
- validation and heldout report hashes where accessed;
- Strong Base, scenario, and protocol hashes;
- selected case identity and paired Strong evidence;
- render attempt ledger;
- video, evidence JSON, board, and method-figure hashes;
- fair-observation, privilege-violation, and test-access fields.

A separate product Showcase audit verifies these fields and representative
video claims. It does not reuse the strict research release audit and cannot
produce an official-test receipt.

## Public presentation

The bilingual README is ordered as:

1. extraction-shooter game loop;
2. four real Bot videos and one comparison board;
3. one-Base-to-three-residual-styles method figure;
4. concise validation evidence with evidence-tier labels;
5. fair-observation boundary;
6. reproduction commands;
7. honest limitations.

The public text states:

- Strong is the shared capable Base;
- Aggressive is a learned directional Showcase admission;
- Explorer is a validation-demonstrated learned style whose heldout
  extraction capability regressed;
- Defensive uses only the strongest evidence tier earned by its final frozen
  calibration;
- no style admission or demonstration is an official-test result.

The README does not claim that extraction rate alone defines Explorer.

## Repository closeout

The final branch:

- tracks the four compact MP4 files, comparison board, method figure, and
  compact evidence manifests;
- removes stale README language saying experiments have not started;
- removes obsolete public v1/v2 media and milestone storytelling while
  retaining reusable implementation needed by v3;
- keeps large checkpoints, raw ledgers, and temporary render attempts outside
  Git;
- records reproducible commands in `script.md`;
- passes targeted tests, full unit tests, Ruff, shell syntax, scenario build
  verification, media audit, public-link checks, and Git cleanliness checks.

The branch is complete for the current product Showcase when all four videos
are visibly representative, every public claim is traceable to a hashed
artifact, the bilingual README is current, and the clean branch is pushed.
The deferred strict research release and official test remain future work.

