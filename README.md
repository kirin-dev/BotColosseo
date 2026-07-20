# Bot Colosseo

Goal-oriented controllable visual game bots via skill-preserving policy shaping.

Bot Colosseo studies how to train a strong visual game Bot and then shape it
into player-recognizable Aggressive, Defensive, and Explorer styles without
discarding its task skill. The approved technical design is in [Plan.md](Plan.md).

## Current status

Milestone 1 is complete. The repository now ships a source-built Crystal Run
scenario, an auditable ACS event protocol, a fair single-agent interface, five
deterministic Teachers, frozen evaluation manifests, and a 500-episode held-out
capability report.

![Crystal Run arena](docs/assets/m1-arena.png)

| Capability | Teacher | Test success | Required |
|---|---|---:|---:|
| Navigation | Fixed Route | 100/100 | ≥95% |
| Pickup | Objective First | 100/100 | ≥95% |
| Return | Evasive Return | 100/100 | ≥95% |
| Static hit | Aggressive Script | 100/100 | ≥90% |
| Moving hit | Aggressive Script | 100/100 | ≥75% |

The official report records zero event-protocol inconsistencies. See the
[Teacher montage](docs/assets/m1-teacher-montage.mp4), [M1 runbook](docs/milestones/m1.md),
and [raw evidence](reports/m1/summary.json).

This is a foundation milestone, not a learned-agent result. M2 learning and
multiplayer, the strong Base Bot, and the learned Aggressive, Defensive, and
Explorer checkpoints are not implemented yet.

## Quick start

```bash
conda env create -f env.yml
conda activate botcolosseo
python scripts/check_env.py
python -m pytest -v
python scripts/build_crystal_run.py --check \
  --acc /home/wencong/.local/bin/acc \
  --acc-include /home/wencong/.local/src/acc-1.60
python scripts/smoke_crystal_run.py \
  --task moving_hit \
  --teacher aggressive_script \
  --record videos/m1-smoke.mp4 \
  --require-video
```

Use `python scripts/evaluate_m1.py --split test --output reports/m1` to reproduce
the full frozen gate. It runs 500 real ViZDoom episodes.

## Licensing

Bot Colosseo source code is MIT licensed. ViZDoom and Freedoom retain their own
licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This repository
does not distribute commercial Doom assets.
