# M4 Showcase Foundation Design

**Date:** 2026-07-22
**Status:** Awaiting written-spec review
**Branch:** `feat/m4-showcase-foundation`

## 1. Purpose

BotColosseo needs a GitHub-native visual proof that a strong visual Bot can be
shaped into a player-recognizable style without discarding its task skill. The
first public vertical slice compares the frozen Strong Base with a learned
Aggressive checkpoint. Defensive and Explorer reuse the same pipeline after M5
training; they are not placeholders in the M4 release.

The showcase is an evidence presenter, not a training or evaluation shortcut.
Every published frame, metric, and label must trace to a real checkpoint, a
frozen validation case, and an episode log.

## 2. Milestone Boundary

The milestone names remain unchanged:

- M0: reliable ViZDoom runtime;
- M1: Crystal Run task and Teacher capability prototype;
- M2: synchronous 1v1 and initial learned Base training;
- M3: historical/PFSP Strong Base gate;
- M4: Aggressive style vertical slice and Base/Aggressive comparison;
- M5: Defensive, Explorer, and difficulty control;
- M6: complete public release and user-facing evidence.

`showcase` is a reusable presentation layer spanning M4 through M6. It is not a
new milestone and does not make a milestone pass. M4 passes only after the
Aggressive style and Skill Retention gates pass in addition to the showcase
artifacts being generated.

## 3. Goals

1. Generate individual MP4s, a compact side-by-side GIF, a metrics card, raw
   episode evidence, and a content-addressed manifest with one command.
2. Compare policies on identical frozen validation conditions.
3. Refuse publication when checkpoints, cases, logs, or claimed gate evidence
   are missing or inconsistent.
4. Keep the README first screen simple enough to understand in roughly 30
   seconds.
5. Allow Defensive and Explorer to be added by configuration rather than new
   rendering code.

## 4. Non-goals

- No web application or interactive frontend.
- No style training, reward design, difficulty controller, or user study in
  this feature.
- No test-split access, official-gate calculation, or manual metric entry.
- No scripted Bot presented as a learned style.
- No empty Defensive or Explorer cards before their real checkpoints pass.
- No publication of the internal M2 PPO/BC pipeline-validation render.

## 5. Public Experience

The M4 README first screen will contain, in order:

1. project title and one-sentence problem definition;
2. a 15--20 second Strong Base/Aggressive GIF from the same case;
3. four evidence-backed numbers: Base performance, Aggressive style shift,
   Skill Retention, and evaluation episode count;
4. one installation command and one demo command;
5. explicit milestone and evidence labels distinguishing validation showcase
   material from official test results.

The GIF uses two learner-perspective columns rather than nesting both duel
perspectives inside each policy panel. Each panel shows only the policy name,
score, core-carrying state, and a key logged behavior label. The subtitle shows
the validation seed, opponent, and side. Full-episode MP4 links sit below the
GIF.

Initial event labels are `PICKUP`, `VALID_HIT`, `DROP`, and `SCORE`. The M4
style evaluator may later supply `ENGAGE` and `CHASE`; the presenter consumes
recorded labels but does not infer style semantics from pixels.

## 6. Architecture

```text
showcase configuration
    + checkpoint files
    + frozen validation cases
    + evaluation metrics artifact
              |
              v
      configuration and hash audit
              |
              v
     matched real ViZDoom episodes
              |
              +--> full episode JSONL
              +--> individual MP4s
              |
              v
 deterministic case/highlight selection
              |
              +--> side-by-side GIF
              +--> metrics card PNG
              +--> publication manifest
```

The implementation adds the following small units:

- `src/botcolosseo/demo/showcase.py`: configuration-independent episode
  capture, learner-frame overlay, frame alignment, and comparison composition;
- `src/botcolosseo/evaluation/showcase.py`: publication configuration,
  eligibility checks, deterministic case/highlight selection, and manifest
  construction;
- `src/botcolosseo/cli/render_showcase.py`: orchestration and atomic output;
- `scripts/render_showcase.py`: thin repository entry point;
- `configs/showcase/development.yaml`: non-public M2 PPO/BC pipeline check;
- `configs/showcase/m4-validation.json`: eight frozen validation showcase
  cases, committed before Aggressive training results are known;
- `configs/showcase/m4.yaml`: added only when the real Strong Base and
  Aggressive publication inputs exist;
- `reports/showcase/m4/`: tracked raw publication evidence;
- `docs/assets/showcase/`: tracked GitHub media.

Training, checkpoint loading, environment execution, and video writing reuse
existing BotColosseo components. The feature does not introduce a second model
loader or duel runtime.

## 7. Configuration Contract

The production configuration is YAML with these required fields:

```yaml
schema_version: 1
stage: m4
publication: true
split: validation
cases: configs/showcase/m4-validation.json
metrics: reports/m4/validation/summary.json
policies:
  - id: strong_base
    label: Strong Base
    checkpoint: runs/m3/selected.pt
    expected_sha256: 0000000000000000000000000000000000000000000000000000000000000001
  - id: aggressive
    label: Aggressive
    checkpoint: runs/m4/aggressive/selected.pt
    expected_sha256: 0000000000000000000000000000000000000000000000000000000000000002
render:
  fps: 10
  gif_seconds: 18
  max_decisions: 525
  output_dir: docs/assets/showcase
evidence_dir: reports/showcase/m4
```

Production policy IDs are restricted by stage: M4 requires exactly
`strong_base` and `aggressive`; M5 permits the addition of `defensive` and
`explorer`. Display labels do not control metric lookup. The metrics artifact
must identify the same policy IDs and checkpoint hashes and must explicitly
report that its source split is validation. The two hashes shown above are
schema examples only; no production configuration containing them is committed.

The development configuration uses `publication: false`, the labels `PPO` and
`BC`, and an ignored artifact output directory. The CLI rejects any attempt to
write a development render under `docs/assets/showcase/` or
`reports/showcase/`.

## 8. Frozen Cases and Deterministic Selection

Eight M4 showcase cases are selected from the existing M3 validation manifest
before Aggressive results are available. They cover both learner sides and the
major opponent behaviors. No test or held-out case is eligible.

After both policies are recorded, a case is publication-eligible only when:

- both episodes terminate normally without truncation;
- both have zero protocol inconsistencies and zero environment retries;
- both demonstrate objective progress;
- checkpoint, scenario, case-manifest, and metric hashes match the config.

The selector ranks eligible cases using an explicitly logged contrast score
from the M4 metrics artifact, with Skill Retention as a hard eligibility check
rather than a compensating term. Ties resolve by case ID. The report retains
all eight case summaries, rejection reasons, the selected case, and the exact
ranking inputs.

Within the selected full episode, an 18-second highlight window is chosen by a
deterministic windowed contrast score over logged behavior events. The full
individual MP4s are always retained so the short GIF cannot hide the rest of
the trajectory. The showcase report labels the GIF as qualitative validation
material; it never substitutes for the complete M4 evaluation.

## 9. Outputs and Provenance

A successful M4 publication produces:

```text
docs/assets/showcase/
  m4-base-vs-aggressive.gif
  m4-strong-base.mp4
  m4-aggressive.mp4
  m4-metrics.png

reports/showcase/m4/
  episodes.jsonl
  case-selection.json
  manifest.json
```

`manifest.json` includes:

- schema and stage;
- Git commit and dirty-worktree state;
- scenario, case-manifest, configuration, metric, checkpoint, episode-log, and
  media SHA-256 values;
- all policy IDs and labels;
- selected case and highlight decision range;
- frame count, dimensions, FPS, and output byte size;
- `split: validation`, `official_test_result: false`, and
  `test_cases_accessed: false`;
- M4 gate identity and pass state imported from the evaluation artifact.

Outputs are written to temporary siblings and atomically renamed only after
all files and hashes validate. A failed run does not overwrite a previous
complete publication.

## 10. Failure Handling

The command exits nonzero without publishing when any of the following occurs:

- a checkpoint is absent or its hash differs;
- the scenario or case manifest drifts;
- a case is not from the frozen validation set;
- the metrics artifact is absent, manually inconsistent, or references other
  checkpoints;
- M4 publication is requested without a passing style/retention gate;
- an episode is empty, malformed, truncated, retried, or protocol-inconsistent;
- no case satisfies the paired eligibility rules;
- a frame stream is empty or has incompatible geometry;
- the GIF exceeds the configured GitHub size ceiling;
- an existing complete output has a different run identity without an explicit
  new output directory.

Errors name the failed artifact and invariant. The renderer never silently
falls back to a script, random policy, different case, or stale media.

## 11. Verification Strategy

Unit tests cover:

- YAML schema, stage-specific policy sets, and path resolution;
- rejection of test cases, bad hashes, manual metric drift, and development
  output in public directories;
- paired eligibility and deterministic tie-breaking;
- highlight-window selection;
- learner-perspective overlay and unequal-length frame alignment;
- empty/incompatible frames and GIF size enforcement;
- canonical manifest hashing and atomic publication behavior;
- public documentation links and absence of machine-specific paths.

A real ViZDoom integration smoke uses the existing local M2 PPO and BC
checkpoints under `publication: false`. It proves that the pipeline can load two
recurrent checkpoints, run the same validation case, write both MP4s, compose a
GIF, and emit a self-consistent manifest. This smoke is short and runs
interactively; it is not a `nohup` experiment.

The final M4 publication gate reruns the same integration path with the frozen
Strong Base and Aggressive checkpoints on CUDA, then audits every hash and
README claim against the produced manifest.

## 12. Acceptance Criteria

The showcase foundation is complete when:

1. the new unit suite and the existing unit suite pass;
2. Ruff and shell/document checks pass;
3. the real M2 development smoke produces valid, non-public media and a
   self-consistent manifest;
4. no M2 development artifact is added to the M4 public asset paths;
5. one documented command is sufficient to generate a production showcase once
   its two real checkpoints and evaluation summary exist;
6. the branch is pushed and opened as a Draft PR without changing the public
   README's milestone claims.

The project is not Resume-ready at this point. Resume-ready remains the later
M4 outcome: a passing Aggressive style/retention gate plus the real published
comparison and updated bilingual README.

## 13. Delivery Sequence

1. Implement and verify the showcase foundation independently of the running
   M3 worktree.
2. Finish and audit M3; freeze the Strong Base checkpoint if its gate passes.
3. Design and implement Aggressive shaping as a separate M4 subsystem.
4. Run the Aggressive experiment and frozen validation gate.
5. Add the production `m4.yaml`, render real assets, and update the README.
6. Reuse the pipeline for Defensive and Explorer during M5.
