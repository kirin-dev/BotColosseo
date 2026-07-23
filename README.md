# Bot Colosseo

[中文说明](README_CN.md) · English

Goal-oriented controllable visual game bots via skill-preserving policy shaping.

Bot Colosseo studies how to train a strong visual game Bot and then shape it
into player-recognizable Aggressive, Defensive, and Explorer styles without
discarding its task skill. The approved technical design is in [Plan.md](Plan.md).

Bot Colosseo 研究如何先训练具备稳定任务能力的视觉游戏 Bot，再在保留能力的
前提下塑造玩家可感知的 Aggressive、Defensive 与 Explorer 行为风格。

## Strong Base → Aggressive

![Strong Base and Aggressive Bot on the same validation case](docs/assets/showcase/m4-base-vs-aggressive.gif)

| Frozen validation evidence | Result |
|---|---:|
| Strong Base win rate | 87.0% |
| Aggressive win rate | 89.0% |
| Aggressive engagement shift | +0.100 / 100 decisions |
| Skill Retention | 100.0% |
| Paired evaluation | 200 episodes |

The Aggressive Bot is a fixed residual-style checkpoint derived from the same
fair-observation Strong Base. It passed all seven predefined style, safety, and
retention gates: the engagement-shift bootstrap interval is `[0.046, 0.171]`,
valid-attack rate is 26.7%, and objective-chase rate is controlled at 9.0%.

This GIF is a qualitative, automatically selected **validation** case—not an
official test result. See the [full Strong Base episode](docs/assets/showcase/m4-strong-base.mp4),
[full Aggressive episode](docs/assets/showcase/m4-aggressive.mp4),
[metric card](docs/assets/showcase/m4-metrics.png), and
[hash-bound publication manifest](reports/showcase/m4/manifest.json).

## Fair Easy / Normal / Hard control

![Difficulty performance on 600 paired validation episodes](docs/assets/showcase/m5-difficulty.png)

The same frozen checkpoints become progressively less restricted from Easy to
Hard. Strong Base performance is `0.820 → 0.878 → 0.955`; Aggressive is
`0.830 → 0.890 → 0.955`. All six frozen gates passed with zero retries and
zero protocol inconsistencies. Hard is the native policy; Normal adds one
decision of reaction delay, while Easy adds two decisions of delay and updates
the policy every two decisions.

This is the passing Strong Base/Aggressive controller calibration, not the
complete M5 claim. Defensive and Explorer still need to pass their style gates
and the frozen all-style difficulty extension. See the
[evidence record](docs/milestones/m5-difficulty.md).

## Current status

Milestone 1 passed its frozen capability gate. Milestone 2 delivered a real
synchronous 1v1 environment, a fair-observation recurrent Actor, 120,000
demonstration transitions, pure-BC initialization, and a 1,000,000-step PPO
run. Its official 1,500-game paired test is complete and integrity-clean, but
the frozen capability gate did **not** pass: PPO clearly beat RandomLegal but
did not improve enough over the strong BC baseline. No M2 capability-pass claim
is made here.

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

| Official M2 test (500 games/policy) | Win rate | Objective rate |
|---|---:|---:|
| PPO | 77.0% | 93.2% |
| Behavioral cloning | 75.2% | 97.8% |
| RandomLegal | 34.4% | 22.0% |

The complete paired rows and frozen gate decisions are tracked in
[`reports/m2/`](reports/m2/). Historical-opponent/PFSP training is now the M3
route for testing whether robustness can improve beyond this M2 plateau. That
run completed with clean integrity evidence but did not pass every frozen M3
capability threshold; its selected 200k checkpoint is therefore described as
an integrity-qualified capability anchor, not an official M3 pass.

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

The learned Aggressive checkpoint and its M4 validation gate are complete. The
[Defensive experiment](docs/milestones/m5-defensive.md) completed its
engineering and paired-validation path but did not produce a statistically
stable style shift, so it is preserved as a negative result rather than called
a successful style Bot. The [Explorer experiment](docs/milestones/m5-explorer.md)
passed its data and offline-learning gates but produced no closed-loop flank
completion in the fixed validation smokes, so it is also retained as a
mechanism-level negative result rather than called complete. Difficulty control
passed its 600-episode Strong Base/Aggressive calibration; the final all-style
M5/M6 product gate remains pending.

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
