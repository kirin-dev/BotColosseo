#!/usr/bin/env bash
# Run from the artifact root with this worktree's src in PYTHONPATH.
set -euo pipefail
task_python=${BOTCOLOSSEO_PYTHON:-python}
task_output=reports/hierarchical/paired-runtime-20261006
mkdir -p "$task_output"
trap 'task_status=$?; echo "evaluation_exit_code=$task_status"' EXIT
for task_style in neutral aggressive defensive explorer; do
  task_args=(--only-style neutral)
  if [[ "$task_style" != neutral ]]; then
    task_args=(--switch --switch-mode style --single-switch-style "$task_style")
  fi
  "$task_python" -u -m botcolosseo.cli.evaluate_hierarchical_styles \
    --executor runs/hierarchical/difficulty-anchored-pilot/epoch-3.pt \
    --strategy runs/hierarchical/random-switch-difficulty10k/last.pt \
    --opponent runs/hierarchical/difficulty-anchored-pilot/control-bundle/strategy-1.pt \
    --record-trace --difficulty 1 --device cpu \
    --seeds 104 105 106 107 108 109 110 111 \
    "${task_args[@]}" --output "$task_output/$task_style.json"
done
