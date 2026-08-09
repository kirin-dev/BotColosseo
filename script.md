# BotColosseo runbook

Run commands from the repository root in the project environment:

```bash
conda activate botcolosseo
export BOTCOLOSSEO_PYTHON="$CONDA_PREFIX/bin/python"
export PYTHONPATH="$PWD/src"
```

The repository contains no machine-specific Conda path. `BOTCOLOSSEO_GPU`
selects a physical GPU; the process sees the selected device as `cuda:0`.

## Verify the release

```bash
"$BOTCOLOSSEO_PYTHON" -m pip check
"$BOTCOLOSSEO_PYTHON" -m ruff check src tests scripts
"$BOTCOLOSSEO_PYTHON" -m pytest -q tests/unit
"$BOTCOLOSSEO_PYTHON" -m pytest -q tests/integration
```

To verify a clean WAD rebuild, also provide ACC:

```bash
"$BOTCOLOSSEO_PYTHON" scripts/build_crystal_run_extraction.py \
  --check \
  --acc /path/to/acc \
  --acc-include /path/to/acc/include
```

## Rebuild mask-aware Teacher data and BC

```bash
"$BOTCOLOSSEO_PYTHON" -m botcolosseo.cli.generate_extraction_demonstrations \
  --config configs/extraction/randomized/aligned-v2/demonstrations.yaml \
  --split train --style strong

"$BOTCOLOSSEO_PYTHON" -m botcolosseo.cli.generate_extraction_demonstrations \
  --config configs/extraction/randomized/aligned-v2/demonstrations.yaml \
  --split validation --style strong

"$BOTCOLOSSEO_PYTHON" -m botcolosseo.cli.train_extraction_bc \
  --config configs/extraction/randomized/aligned-v2/bc.yaml \
  --style strong --device cuda:0
```

Generation and training are resumable. Do not reuse partial data after changing
the scenario, layout generator, Teacher, cases, or configuration.

## Reproduce the released Strong run

Start with a 10k admission slice:

```bash
BOTCOLOSSEO_PYTHON="$BOTCOLOSSEO_PYTHON" \
  bash scripts/launch_conservative_strong_ppo_10k.sh
```

After checking the log and 10k checkpoint, resume the same optimizer to the
configured 1M budget:

```bash
BOTCOLOSSEO_PYTHON="$BOTCOLOSSEO_PYTHON" \
  bash scripts/resume_conservative_strong_ppo_1m.sh
```

The run writes candidates every 50k steps. Candidate screening is performed
after training; the released model is the 950k checkpoint, not the final one.
The configuration uses BC replay, reference KL, scheduled Teacher supervision,
and `history_probability: 0.0`.

## Reproduce style adapters

Aggressive:

```bash
BOTCOLOSSEO_PYTHON="$BOTCOLOSSEO_PYTHON" \
BOTCOLOSSEO_STYLE_STEPS=100000 \
  bash scripts/run_extraction_randomized_opportunity_styles.sh
```

The released Defensive and Explorer repairs use:

```bash
BOTCOLOSSEO_PYTHON="$BOTCOLOSSEO_PYTHON" BOTCOLOSSEO_GPU=0 \
  bash scripts/run_extraction_randomized_disengagement_style.sh

BOTCOLOSSEO_PYTHON="$BOTCOLOSSEO_PYTHON" \
  bash scripts/run_extraction_randomized_aligned_opportunity_styles.sh
```

These scripts train bounded residual adapters over the same frozen 950k Strong
Actor. They do not read official-test cases.

## Evaluate

```bash
"$BOTCOLOSSEO_PYTHON" -m botcolosseo.cli.evaluate_extraction_v3 \
  --checkpoint runs/extraction-randomized/strong-ppo-conservative-v2/candidate-0950000.pt \
  --policy strong \
  --protocol configs/extraction/randomized/evaluation.yaml \
  --split validation \
  --device cuda:0 \
  --output /tmp/botcolosseo-strong-validation.json
```

Use `--base-checkpoint` when evaluating a style adapter. Keep validation,
heldout, and sealed official-test outputs separate.

## Rebuild and audit the Showcase

```bash
BOTCOLOSSEO_PYTHON="$BOTCOLOSSEO_PYTHON" \
  bash scripts/render_extraction_v3_showcase.sh

"$BOTCOLOSSEO_PYTHON" -m botcolosseo.cli.audit_extraction_showcase \
  --selection reports/extraction/showcase/selection.json \
  --board-manifest reports/extraction/showcase/manifest.json \
  --method docs/assets/extraction/method.svg \
  --experiment-identity reports/extraction/showcase/experiment-identity.json \
  --output /tmp/botcolosseo-showcase-audit.json
```

The audit verifies the selected cases, evidence tiers, checkpoint bindings,
method diagram, board, video hashes, and composite experiment identity.
