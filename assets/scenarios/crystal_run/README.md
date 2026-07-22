# Crystal Run Arena

This directory contains the source and runnable artifact for Bot Colosseo's
Crystal Run scenario. MAP01–MAP06 retain the M1 single-instance tasks; MAP07 is
the version-2 duel map introduced in M2.

- `src/map.udmf` is the reviewable arena geometry.
- `src/crystal_run.acs` owns task setup, protocol-v1 task counters, and the
  backwards-compatible protocol-v2 duel block.
- `src/regions.yaml` defines stable semantic regions and Teacher routes.
- `src/task_variants.yaml` maps tasks to WAD map markers.
- `crystal_run.wad` is generated and tracked so users can run the scenario
  without installing ACC.
- `manifest.json` binds the artifact to its sources and compiler version.

Rebuild from the repository root:

```bash
python scripts/build_crystal_run.py \
  --acc /path/to/acc \
  --acc-include /path/to/acc/source
```

SLADE is optional for visual inspection and is not part of the headless build.
