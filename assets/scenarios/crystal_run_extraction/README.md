# Crystal Run: Extraction scenario

This directory contains the source-built protocol-v3 ViZDoom scenario used by
the project.

The WAD supports two rule-identical loot layouts:

- `crystal_run_extraction.cfg`: `base`, used for training and validation;
- `crystal_run_extraction_heldout.cfg`: `heldout-a`, used only for heldout and
  the frozen official test.

Build and verify:

```bash
export ACC_ROOT=/path/to/acc/source
PYTHONPATH=src python scripts/build_crystal_run_extraction.py \
  --check \
  --acc "$ACC_ROOT/build/acc" \
  --acc-include "$ACC_ROOT"
```

A tracked WAD alone is not experimental evidence. Mechanics are covered by
deterministic rule tests and real synchronous two-player ViZDoom preflights.
