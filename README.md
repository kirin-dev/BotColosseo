# Bot Colosseo

Goal-oriented controllable visual game bots via skill-preserving policy shaping.

Bot Colosseo studies how to train a strong visual game Bot and then shape it
into player-recognizable Aggressive, Defensive, and Explorer styles without
discarding its task skill. The approved technical design is in [Plan.md](Plan.md).

## Current status

Milestone 0 verifies reproducible headless ViZDoom initialization, deterministic
actions, episode termination/reset, and MP4 recording. The Crystal Run scenario,
learning pipeline, multiplayer, and style models are planned but are not claimed
as implemented.

## Quick start

```bash
conda env create -f env.yml
conda activate botcolosseo
python scripts/check_env.py
python -m pytest -v
python scripts/smoke_vizdoom.py \
  --episodes 1 \
  --record videos/m0-smoke.mp4 \
  --require-video
```

See [the M0 runbook](docs/milestones/m0.md) for acceptance criteria and
troubleshooting.

## Licensing

Bot Colosseo source code is MIT licensed. ViZDoom and Freedoom retain their own
licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This repository
does not distribute commercial Doom assets.
