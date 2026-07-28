# Aggressive Style Calibration Design

## Status

Approved bounded calibration for the `feat/crystal-run-extraction-v3`
branch. It preserves the frozen game, Strong Base, validation protocol, and
public learned-policy boundary.

## Evidence and problem

The original Aggressive run improved its paired style signal across the fixed
240-case validation set:

| Checkpoint | Style difference | 95% CI lower | Task retention |
| --- | ---: | ---: | ---: |
| 200k | +0.0229 | -0.0970 | 0.9112 |
| 400k | +0.1027 | -0.0226 | 0.9346 |
| 600k | +0.1009 | -0.0090 | 0.9346 |

At 600k, ten of eleven validation checks pass. The sole failure is the
predefined requirement that the paired style-difference confidence interval
have a positive lower bound. The optimizer learning rate has decayed to
approximately zero, so extending the same schedule is not justified.

## Calibration

Calibration starts from the 600k Aggressive model weights while resetting the
optimizer, scheduler, environment-step counter, and rollout state. It:

- keeps the exact Strong Base checkpoint frozen;
- keeps the task reward, KL coefficient, residual penalty, action space, and
  observation boundary unchanged;
- scales the complete Aggressive reward ledger from `1.0` to `1.5`, including
  its invalid-attack and low-resource-attack penalties;
- uses a fresh low learning-rate schedule;
- trains to 100k environment steps first and may resume to 200k only if the
  fixed validation gate still fails;
- never reads heldout cases unless a candidate passes validation, and never
  reads official-test cases during calibration or selection.

This is weights-only initialization, not checkpoint resume. The output records
the parent checkpoint path and SHA-256 so that the lineage cannot be confused
with an uninterrupted run.

## Artifact isolation and promotion

The original 200k, 400k, and 600k artifacts remain unchanged under
`runs/extraction/styles/aggressive/`.

Calibration artifacts are written under
`runs/extraction/styles/aggressive-calibration-v2/`. Candidate evaluation uses
the same frozen 240-case validation protocol and unchanged gates. A passing
candidate must also pass the existing 120-case heldout gate.

Only a fully eligible calibrated selection may be promoted into the canonical
Aggressive location. Promotion copies the selected checkpoint and selection
manifest without changing their hashes, retains evidence paths into the
calibration directory, and refuses to overwrite an existing eligible
canonical selection.

## Implementation boundary

The style-training CLI gains an explicit weights-only initialization option
that:

1. validates the parent checkpoint schema, scenario hash, base checkpoint
   hash, and style identity;
2. loads model weights but no optimizer, scheduler, counters, or RNG state;
3. writes immutable parent lineage into checkpoints and the final summary;
4. remains mutually exclusive with normal `--resume`.

The runners gain isolated output and promotion support. Unit tests cover
argument exclusivity, lineage validation, fresh counters, output isolation,
and fail-closed promotion. Existing resume behavior remains unchanged.

## Success criteria

The calibration is successful only when:

- the validation paired style CI lower bound is greater than zero;
- all existing task-retention, capability, anti-hacking, and protocol checks
  pass unchanged;
- heldout gates pass;
- `test_cases_accessed` remains false;
- the promoted checkpoint hash equals the selected calibration checkpoint
  hash;
- the released policy remains a learned residual policy with no runtime
  governor.

If the 200k calibration candidate still fails, this design stops. It does not
authorize further reward scaling, gate relaxation, scene changes, or
inference-time scripting.
