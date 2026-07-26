# Crystal Run Extraction v2

This is the isolated protocol-v3 implementation of the minimal 1v1 extraction
scenario. It reuses the compact Crystal Run arena geometry but has independent
rules, observations, splits, evidence, and checkpoints.

Build it with:

```bash
PYTHONPATH=src python scripts/build_crystal_run_extraction.py \
  --acc /home/wencong/.local/bin/acc \
  --acc-include /home/wencong/.local/src/acc-1.60
```

The tracked WAD is not evidence of a passed mechanics gate by itself. The X0
gate also requires deterministic rule tests and a real synchronous ViZDoom
replay covering loot, death, corpse-cache transfer, and extraction.
