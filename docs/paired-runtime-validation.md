# Paired runtime-control validation

Purpose: validate the deployed policy's response to a mid-match style change,
then publish an explanatory comparison for the showcase and interview handoff.

## Frozen protocol

- Same selected high Actor, anchored executor and Neutral/Hard opponent as the
  current showcase. No training or checkpoint selection in this validation.
- Layout seeds 104–111, both learner roles: 16 cases, four arms, 64 games.
- All arms start Neutral/Hard. Control arm stays Neutral. Intervention arms
  switch once at decision 81 to Aggressive, Defensive or Explorer, staying Hard.
- Same per-player sampling seeds across arms. Recurrent memory persists.
- Record both players' frame/self-state hashes and learner action/command traces.
  Check equality through the observation entering decision 81, before comparing
  post-switch behavior. Prefix mismatches invalidate a matched-state comparison;
  keep those outcomes as descriptive seed/role-paired diagnostics.
- Report applied-control latency separately from behavioral response. Compare
  attack actions, search/extraction command occupancy in decisions 88–160;
  report missing/short windows and terminal outcomes rather than discarding them.
- Compare banked value and positive-value extraction per case, with layout-level
  paired bootstrap uncertainty. These shared development cases do not establish
  independent generalization or universal switching benefit.

## Execution and delivery

Run `scripts/run_paired_runtime_controls.sh` from the artifact root with this
worktree's `src` on PYTHONPATH. Preserve incomplete reports if the run fails.
Reports are under `reports/hierarchical/paired-runtime-20261006`.

After completion: audit identities, schedules and common prefixes; summarize
behavior and task outcomes; select a representative matched-prefix case and
render side-by-side videos with identical time axes and control annotations.
Update the current Pages results and bilingual README with supported findings;
update the private BotColosseo.md interview handoff without publishing it.

If response is stable, finish without retraining. If a reproducible visible
failure appears, diagnose it and use a small targeted repair before publishing.

## Observed outcome

All 64 games completed. Each style arm applied all 16 requests, maximum delay
7 decisions. Matching full observation/action prefixes: A 4/16, D 4/16, E 2/16.
The replay engine does not reproduce identical histories for every matched seed.
Aggregate payoff differences therefore do not isolate a causal switching effect.

The selected Defensive video was independently recorded and passes observation
and action/command prefix equality. Both arms bank 45; extraction times are
38.97s and 32.80s. Published `assets/hierarchical/paired-runtime.json` binds raw
reports, video identities, behavior windows, payoff uncertainty and clip hashes.
The read-only same-history probe preserves actual actions, RNG and hidden state.
