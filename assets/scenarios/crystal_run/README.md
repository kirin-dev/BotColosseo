# Crystal Run Arena

This directory contains the source and runnable artifact for Bot Colosseo's M1
single-instance scenario.

- `src/map.udmf` is the reviewable arena geometry.
- `src/crystal_run.acs` owns task setup and protocol-v1 counters.
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
