# M6 Public Release Design

**Status:** Approved through delegated recommendation authority on 2026-07-23.

## 1. Product outcome

M6 turns the existing experiment repository into a compact, evidence-backed
GitHub product demonstration. A visitor should understand within one screen
that BotColosseo starts from one fair-observation Base Bot, derives three
recognizably different policies, and controls difficulty independently.

The release does not add a web application. It reuses the audited showcase
renderer and publishes:

- one 15--20 second 2-by-2 GIF containing Strong Base, Aggressive, Defensive,
  and Explorer on the same frozen validation case;
- one full MP4 per policy;
- one compact evidence card;
- one difficulty result card;
- bilingual project documentation;
- an anonymous blind user-study package;
- hash-bound checkpoint and publication manifests.

## 2. Evidence boundary

M6 presentation is downstream of M4/M5 evidence, never an alternate gate.

- Aggressive must retain its passing M4 validation identity.
- Defensive and Explorer may enter the final four-policy release only through
  passing closed-loop PPO evidence with the unchanged evaluators.
- Difficulty claims require the passing 1,800-episode all-style M5 audit. The
  already-passing 600-episode Base/Aggressive block is one of its three inputs.
- A failed candidate remains documented as a negative result. It may be
  rendered locally for diagnosis but is never labelled as a completed style
  in the public M6 publication.
- Every public asset records `split: validation`,
  `official_test_result: false`, and `test_cases_accessed: false`.

If one repair does not pass, the existing M4 Base/Aggressive public showcase
stays canonical while the failed M5 evidence remains visible. M6 is not called
complete by weakening the publication contract.

## 3. Chosen presentation

### 3.1 Four-policy comparison

The existing horizontal two-policy compositor is generalized:

- two streams retain the current one-row layout;
- three or four streams use a deterministic 2-by-2 grid;
- an empty fourth panel is used only for a three-stream non-public diagnostic;
- panel geometry, overlay contents, frame alignment, subtitle, and GIF byte
  ceiling remain unchanged.

A grid is more legible than a 1,024-pixel-wide row in GitHub and avoids a new
frontend. The selected case must be eligible for all four policies. The short
window is deterministic; full episodes remain linked beside it.

### 3.2 Metrics

M6 uses a new versioned metric payload rather than overloading the
Aggressive-only M4 schema. It contains:

- Strong Base win rate;
- Aggressive engagement shift and Skill Retention;
- Defensive protective-presence shift and Skill Retention;
- Explorer route-entropy shift and Skill Retention;
- difficulty monotonicity and objective-capability result;
- episode counts, estimator names, checkpoint hashes, and upstream summary
  hashes.

The renderer consumes only display fields and contrast scores. A separate
builder validates each upstream formal artifact and emits the payload, so
numbers cannot be entered manually in the showcase config.

## 4. User study

The project author will conduct the human evaluation after video generation.
The repository supplies a reproducible blind package:

1. clips are renamed to opaque IDs using a committed seed;
2. assignment order is counterbalanced;
3. respondents select Aggressive, Defensive, Explorer, or Unsure for each
   anonymous clip;
4. they rate style clarity and perceived difficulty on a five-point scale;
5. no name, email, IP address, or free-text personal information is collected.

The committed response schema contains only:

```text
respondent_id, assignment_id, clip_id,
style_choice, style_clarity, perceived_difficulty
```

Raw anonymous CSV, assignment manifest, and summary hashes are retained.
Analysis reports sample size, confusion matrix, per-style recognition rate,
macro recognition rate, Wilson intervals, clarity, and perceived difficulty.
With 5--10 respondents, results are explicitly described as a small product
perception study rather than population-level evidence.

## 5. Checkpoint release

Large checkpoints are not committed to normal Git history. A release builder
copies only the four inference checkpoints, strips optimizer and critic state
where supported, records architecture/scenario/source hashes, verifies
fair-observation loading, and emits a content-addressed manifest. GitHub Release
upload remains a separate explicit publication action.

## 6. Implementation order

1. Generalize the comparison compositor to 2-by-2 layouts without changing M4.
2. Add an M6 config/metric schema and provenance builder.
3. Add blind user-study packaging and analysis.
4. Add checkpoint release verification and manifest generation.
5. When M5 results freeze, bind exact hashes and run the real showcase.
6. Conduct the user study, commit anonymous evidence, and finish bilingual
   README and milestone records.

## 7. Verification

Unit tests cover two-panel backward compatibility, four-panel geometry, frame
alignment, M6 policy order, upstream hash/gate rejection, blind assignment
determinism, response validation, confusion-matrix calculations, checkpoint
manifest integrity, and zero-test-access claims.

The production render runs only from a clean Git commit and refuses partial,
failed, hash-mismatched, test-derived, or manually edited evidence.
