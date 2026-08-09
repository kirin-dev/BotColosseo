# BotColosseo

**Controllable Game Bots for Search-Fight-Extract**

[中文说明](README_CN.md)

One capable first-person ViZDoom Bot, three learned play styles, and one compact
1v1 extraction task.

## [Open the interactive Showcase →](https://kirin-dev.github.io/BotColosseo/)

Watch four directly playable videos, inspect the arena, understand the training
flow, compare the styles, and read the most relevant capability evidence.

## What is the task?

```text
search for loot → fight or disengage → manage inventory → extract → bank value
```

- One 75-second 1v1 raid with two neutral extraction zones.
- Each player has **100 HP**; every valid hit deals **20 damage**.
- Each player starts with **30 rounds**, with no reload or respawn.
- The backpack has **three slots** for loot worth 10, 25, or 50.
- Death drops all unbanked loot into a collectible corpse cache.
- A kill has no intrinsic score; only extracted value counts.

## One base, four visible behaviors

| Bot | Priority | Representative causal chain |
|---|---|---|
| **Strong** | balanced task capability | search → valuable loot → extract → bank |
| **Aggressive** | useful combat conversion | hit → kill → corpse cache → extract |
| **Defensive** | preserve carried value under risk | stop pursuit → disengage → extract |
| **Explorer** | useful route and loot diversity | search regions → upgrade backpack → extract |

The Strong CNN-GRU Actor is trained through scripted Teacher data, behavioral
cloning, recurrent PPO, historical opponents, and lightweight PFSP. The three
styles are bounded residual logit adapters over the same frozen Strong Actor hash.
Their reward shaping is activated by training-only opportunity detectors; the
deployed policies remain learned residual adapters with the same public inputs.

Closeout auditing fixed PFSP draw bookkeeping after the frozen Strong checkpoint
was trained. Its closed-loop evaluation below is unchanged, but this release
does not claim a causal PFSP gain; retraining is deferred.

### Current v3 code path

| Layer | Entry point |
|---|---|
| Game rules and ACS map | `assets/scenarios/crystal_run_extraction_randomized/` |
| Synchronized environment and public protocol | `src/botcolosseo/envs/synchronous_extraction.py` |
| CNN-GRU Actor and asymmetric Critic | `src/botcolosseo/agents/extraction_model.py` |
| BC, recurrent PPO, PFSP, and style shaping | `src/botcolosseo/training/extraction_*.py` |
| Frozen evaluation and evidence tiers | `src/botcolosseo/evaluation/extraction_*.py` |
| Reproducible commands | `script.md` |

## Results

| Strong capability | Result |
|---|---:|
| Randomized validation extraction | **83.3%** |
| Randomized validation win rate | **56.7%** |
| Randomized validation mean banked value | **39.10** |
| Randomized heldout extraction | **85.8%** |

### Randomized-layout release

The released Strong and all three style adapters share the same randomized-loot
scenario and frozen Strong checkpoint. Seven loot items are assigned across 16
safe anchors through finite, collision-free permutations. This is domain
randomization over a finite layout family, not continuous-placement
generalization. Frozen 32-episode screens selected the 950k checkpoint; the
public capability numbers above come from a separate 240-episode confirmation.
See the [derived curve data](reports/extraction/training-curve.json).

| Bot | Public evidence | What the selected video proves |
|---|---|---|
| Aggressive | Representative case | 5 hits → kill → corpse cache → 85-value extraction |
| Defensive | Representative case | carried-value disengagement → extraction, 0 kills |
| Explorer | Representative case | 4 loot regions → backpack upgrade → 85-value extraction |

These are validation-selected product demonstrations, not proof that every
style improves on the full distribution. Each video is explicitly scoped as a
representative case study with a complete, engine-recorded causal chain. See the
[machine-readable audit](reports/extraction/showcase/audit.json) for cases,
checkpoint/media hashes, evidence tiers, and every disclosed failed check.

## Evidence boundary

The deployed Actor sees only its 84×84 first-person grayscale frame and its own
public state. The Actor never receives opponent HP or position, world
coordinates, depth, labels, automap, or viewer telemetry. Privileged state is
confined to the asymmetric training Critic and reward shaping, plus offline
evaluation and viewer telemetry; none of it enters the deployed Actor.

This is a product Showcase with no benchmark-success claim and no aggregate
skill-preservation claim for all styles.
Candidate selection never accesses the test split. The deferred protocol allows
one frozen 400-episode official test per policy (1,600 episodes total); it has
not been run.

## Reproduce

Use the `botcolosseo` Conda environment. Full commands are in
[script.md](script.md), and frozen gates are in [Plan.md](Plan.md).

```bash
conda activate botcolosseo
python -m pip check
python -m ruff check src tests scripts
python -m pytest tests/unit -q
```

Project source is MIT licensed. ViZDoom and Freedoom retain their respective
licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
