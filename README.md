# Bot Colosseo

Goal-oriented controllable visual game bots via skill-preserving policy shaping.

Bot Colosseo studies how to train a strong visual game Bot and then shape it
into player-recognizable Aggressive, Defensive, and Explorer styles without
discarding its task skill. The approved technical design is in [Plan.md](Plan.md).

## Current status

Milestone 1 is complete. Milestone 2 now has a real synchronous 1v1 environment,
a fair-observation recurrent Actor, 120,000 demonstration transitions, pure-BC
initialization, and a 1,000,000-step PPO run. The official 1,500-game paired M2
test is frozen and prepared but has not yet been run, so no PPO-over-BC test
claim is made here.

![M2 fair-observation learning system](docs/assets/m2-system.png)

![M2 validation-only training evidence](docs/assets/m2-training-curves.png)

| M2 training artifact | Validation evidence | Selected checkpoint |
|---|---:|---:|
| Behavioral cloning | 95.61% action accuracy | update 5,500 |
| Recurrent PPO | 100% objective rate, 83.33% win rate (30 games) | 800k steps |

These are validation-only selection numbers. See the
[PPO-versus-BC validation showcase](docs/assets/m2-policy-comparison.mp4),
[M2 evidence record](docs/milestones/m2.md), and tracked
[training summaries](reports/m2/). The showcase uses the selected checkpoints
on one frozen validation seed; it is qualitative rather than an official
performance sample.

Milestone 1 established the source-built Crystal Run scenario, auditable ACS
event protocol, fair single-agent interface, five deterministic Teachers,
frozen evaluation manifests, and a 500-episode held-out capability report.

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

The M3 Strong Base/PFSP gate, difficulty control, and the learned Aggressive,
Defensive, and Explorer checkpoints remain pending.

## Quick start

```bash
conda env create -f env.yml
conda activate botcolosseo
python scripts/check_env.py
python -m pytest -v
ACC_PATH=/path/to/acc ACC_INCLUDE=/path/to/acc/source \
  python scripts/build_crystal_run.py --check
python scripts/smoke_crystal_run.py \
  --task moving_hit \
  --teacher aggressive_script \
  --record videos/m1-smoke.mp4 \
  --require-video
python scripts/plot_m2_training.py
```

Use `python scripts/evaluate_m1.py --split test --output reports/m1` to reproduce
the full frozen M1 gate. It runs 500 real ViZDoom episodes. M2 commands and the
strict official artifact audit are documented in the
[M2 evidence record](docs/milestones/m2.md) and [`script.md`](script.md).

## Licensing

Bot Colosseo source code is MIT licensed. ViZDoom and Freedoom retain their own
licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This repository
does not distribute commercial Doom assets.
