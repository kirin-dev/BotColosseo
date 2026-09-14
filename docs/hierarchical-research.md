# Hierarchical control research

[Watch the research preview](https://kirin-dev.github.io/BotColosseo/hierarchical.html).
This is a separate research route; the original residual-adapter Showcase remains available.

## Architecture

A low-frequency recurrent Actor selects seven commands: search north/center/south,
engage, disengage, or extract at either exit. A shared visual CNN–GRU executor
acts every four engine tics. High-level replanning occurs every eight low decisions,
or earlier after public events. Each player has independent recurrent memory.

Style conditions the high Actor through FiLM. Difficulty conditions both layers;
the optional Hard anchor freezes the reference transform while learning relative
low-difficulty modulation. Runtime control changes do not swap weights or reset memory.
Actors receive only first-person observations and public own state. Privileged
state is restricted to training supervision, value targets and viewer diagnostics.

## Code map

| Component | Module under `src/botcolosseo/` |
|---|---|
| Models and inference | `agents/hierarchical_model.py`, `agents/hierarchical_controller.py` |
| Command Teacher | `agents/hierarchical_teacher.py` |
| Command BC and conservative PPO | `training/hierarchical_bc.py`, `cli/train_hierarchical_ppo.py` |
| Strategic SMDP PPO | `training/hierarchical_collection.py`, `training/hierarchical_strategic_ppo.py` |
| General-sum game and versioned population | `training/hierarchical_game.py`, `training/hierarchical_population.py` |
| Style/difficulty distillation | `cli/distill_hierarchical_styles.py`, `cli/distill_hierarchical_difficulty.py` |
| Live control and recording | `demo/hierarchical_controls.py`, `cli/render_hierarchical.py` |

## Entry points

Install the repository's `training` and `dev` extras, then run:

```bash
python -m pytest tests/unit/test_hierarchical_*.py -q
python -m botcolosseo.cli.probe_hierarchical_teacher --help
python -m botcolosseo.cli.train_hierarchical_bc --help
python -m botcolosseo.cli.run_hierarchical_matrix --help
python -m botcolosseo.cli.train_hierarchical_strategic --help
python -m botcolosseo.cli.evaluate_hierarchical_styles --help
python -m botcolosseo.cli.render_hierarchical --help
```

Training checkpoints and trajectory archives are not bundled with this source
release. Training and rollout commands require locally generated artifacts;
the CLI arguments expose their paths. The public preview includes selected
videos and a compact [evidence summary](assets/hierarchical/evidence.json).

## Interpretation

The empirical game uses each player's own extracted value divided by 150, not
opponent-denial reward. Both role payoff matrices are retained. Meta-strategies
are sampled before episodes; shared-executor changes require a new game identity.
Independent comparisons and layout-cluster bootstrap prevent a small matrix's
zero regret from being mistaken for global convergence or reliable improvement.

Cross-play accepts `--condition A D E DIFFICULTY` (default `0 0 0 1`). Both
players use that fixed condition; it is bound into pair, matrix and solution
identities. Different conditions cannot share resumed games or bootstrap cells.
Existing Neutral-only response/upgrade/comparison tools reject non-Neutral games;
conditioned population training is not implied by this cross-play interface.

Two response rounds and an executor-update experiment are implemented, but their
independent results do not establish a reliable response/meta-strategy gain.
Runtime style/difficulty inputs work in continuous episodes; a consistently
monotonic difficulty ladder and population-wide conditional-response validation
remain open. The style showcase is conditional distillation, not a claim that
named styles emerged spontaneously from PSRO.
