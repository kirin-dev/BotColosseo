# BotColosseo

### Controllable Game Bots for Search-Fight-Extract

**One policy. Three play styles. Real-time control.**

[**Watch the showcase →**](https://kirin-dev.github.io/BotColosseo/) · [中文](README_CN.md) · [Release](https://github.com/kirin-dev/BotColosseo/releases/tag/v0.1.0) · [Code guide](docs/hierarchical-research.md)

A first-person ViZDoom bot that changes its priorities during a match—without
swapping model weights or resetting memory. A high-level planner chooses goals;
a shared visual executor turns them into actions.

## See it in action

| Live controls | Aggressive |
|:---:|:---:|
| [![Live control episode](docs/assets/hierarchical/curriculum-live.jpg)](https://kirin-dev.github.io/BotColosseo/#styles) | [![Aggressive episode](docs/assets/hierarchical/curriculum-aggressive.jpg)](https://kirin-dev.github.io/BotColosseo/#top) |
| Change style + difficulty → extract **60** value | Eliminate threat → search loot → extract **85** value |
| **Defensive** | **Explorer** |
| [![Defensive episode](docs/assets/hierarchical/curriculum-defensive.jpg)](https://kirin-dev.github.io/BotColosseo/#top) | [![Explorer episode](docs/assets/hierarchical/curriculum-explorer.jpg)](https://kirin-dev.github.io/BotColosseo/#top) |
| Acquire value → seek extraction → bank **50** | Search → three pickups → extract **85** value |

Selected cases, not average performance. All four share the same high-level Actor,
executor and frozen opponent. [Video identities and events](docs/assets/hierarchical/curriculum-showcase.json)

## The game

Search for loot → fight or disengage → extract → bank value.

- **75-second 1v1 raid**, two neutral exits; both players may extract.
- **100 HP**, **20 damage** per valid hit, **30 rounds** initially; no reload or respawn.
- **Three slots** for loot worth 10 / 25 / 50; better loot automatically replaces the lowest-value item.
- Death drops unbanked loot. Only your own extracted value earns task reward—kills are optional.

Geometry is fixed; seven loot items vary across sixteen anchors within a finite
layout family. This is randomized loot, not procedurally generated maps.

## How it works

![High-level planner and shared low-level executor](docs/assets/hierarchical/method.svg)

| Component | Responsibility |
|---|---|
| **High-level planner** | Recurrent GRU selects search regions, engage/disengage or either exit: seven commands. |
| **Shared executor** | CNN–GRU maps first-person observations and the command to movement, turning and fire. |
| **Runtime controls** | Bounded FiLM injects style into the planner and difficulty into both layers; weights and memory persist. |

**Training:** Teacher demonstrations → command BC / conservative executor PPO →
conditional high-level distillation → random-segment style/difficulty PPO.
Preference targets guide initialization; styles are not claimed to emerge from PSRO.

The Actor uses first-person pixels, public own state and history. Hidden enemy
state and viewer telemetry are not policy inputs; privileged supervision and
Critic inputs stay on the training side.

## Measured behavior

**Difficulty · 192 development games · one frozen deployment policy**

| Input | Mean banked value | Positive-value episodes |
|---|---:|---:|
| Easy | **21.33** | 67.19% |
| Normal | **32.34** | 81.25% |
| Hard | **38.20** | 85.94% |

**Style · same early post-switch window · Hard difficulty**

| Decision-step occupancy | Aggressive | Defensive | Explorer |
|---|---:|---:|---:|
| Attack actions | **8.43%** | 0.00% | 0.21% |
| Search commands | 78.94% | 52.23% | **95.21%** |
| Extraction commands | 4.11% | **47.77%** | 4.79% |

The style diagnostic covers six switch orders and 96 episodes, reusing 16
layout/role cases. Occupancy is not a success rate. These are development results:
not every style has monotonic difficulty, and switching benefits or independent
generalization are not established.

[Difficulty data](docs/assets/hierarchical/task-difficulty.json) · [Style data](docs/assets/hierarchical/counterbalanced-styles.json) · [Full release scope](docs/hierarchical-release.md)

## Run the code

Python 3.10 is required. Install the appropriate PyTorch build for your machine.

```bash
python -m pip install -e ".[training,dev]"
python -m pytest tests/unit -q
python -m botcolosseo.cli.evaluate_hierarchical_styles --help
python -m botcolosseo.cli.render_hierarchical --help
```

This is a **source-and-showcase release**. Pretrained checkpoints and raw training
trajectories are not bundled; rollout commands require locally generated artifacts.
See the [architecture and entry points](docs/hierarchical-research.md).

<details>
<summary>Project evolution & research boundaries</summary>

Fixed-loot bots → randomized-loot residual styles → shared hierarchical runtime
control. Future VLM planning and human-likeness are directions, not delivered features.

PSRO and executor-update experiments exist, but reliable response gains and formal
executor promotion were not established. No claim is made that all research gates passed.

The earlier Strong/residual-adapter metrics belong to a different deployment:
[historical baseline](docs/adapter-baseline.md) · [original videos](https://kirin-dev.github.io/BotColosseo/adapter.html).

</details>

---

MIT source license. ViZDoom and Freedoom retain their own licenses;
see [third-party notices](THIRD_PARTY_NOTICES.md).
