# Hierarchical runtime-control showcase release

[Open the showcase](https://kirin-dev.github.io/BotColosseo/)
· [Architecture and code](hierarchical-research.md)

## The product problem

A game bot should change its priorities during a match without loading another
policy. BotColosseo separates low-frequency goal selection from shared visual
execution, and supplies style and difficulty as runtime conditions.

The environment is a deliberately small search-fight-extract task, not a complete
commercial extraction game. Two players search randomized loot locations, manage
three inventory slots and decide whether combat is worth the risk. Killing is
optional; each player optimizes their own extracted value. Both may extract.
The reported winner is a secondary comparison of final values, not the training
objective. Automatic replacement of the least valuable inventory item is an
environment rule, not a learned inventory-management action.

## What is delivered

- One fixed high-level Actor plus one shared executor, with persistent recurrent
  memory and runtime FiLM controls. This is one deployment combination, not one
  monolithic network and not separate model weights for each style.
- Four engine-recorded cases: Aggressive eliminates an opponent and banks 85;
  Defensive banks 50; Explorer banks 85 after three pickups; live controls change
  during one episode that banks 60. The Aggressive case does not loot a corpse cache.
- A 192-game development evaluation: mean banked value **21.33 / 32.34 / 38.20**
  for Easy / Normal / Hard. Positive-value episode rates are **67.19% / 81.25% /
  85.94%**. Empty-handed extraction is a separate metric.
- Six-order runtime-style diagnostics: 96 episodes, eight layouts and both roles.
  In the same first post-switch window, Aggressive attack occupancy is 8.43%,
  Defensive extraction-command occupancy 47.77%, and Explorer search-command
  occupancy 95.21%. The showcase compares every style on all three metrics.

## Evidence and boundaries

The [video manifest](assets/hierarchical/curriculum-showcase.json),
[difficulty results](assets/hierarchical/task-difficulty.json) and
[style diagnostics](assets/hierarchical/counterbalanced-styles.json) bind the
same Actor, executor and opponent hashes. Original adapter-release numbers are
separate and remain available from the original showcase.

Actors use first-person pixels, public own state and recurrent history. Teacher,
Critic and training rewards may use privileged state; viewer-only enemy telemetry
is not a policy input. Style initialization uses demonstrations and explicit
preference targets, followed by conditional distillation and runtime-segment PPO;
styles did not emerge spontaneously from PSRO.

The arena geometry is fixed and loot varies over a finite layout family. Results
are development evidence, not unseen-map generalization, human-rated difficulty
calibration, or proof that changing controls increases returns. Repeated orders
are not independent samples, trajectories with matched seeds are not identical,
and not every style follows a monotonic difficulty curve. Formal skill retention,
reliable PSRO response gains and executor promotion remain unestablished.

This release includes source, videos and evidence summaries. It does not bundle
pretrained checkpoints or raw training trajectories; training/evaluation commands
require local artifacts. Future VLM planning, human-likeness and automatic DDA
are extensions, not delivered capabilities.
