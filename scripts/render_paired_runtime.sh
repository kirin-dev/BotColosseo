#!/usr/bin/env bash
set -euo pipefail
task_python=${BOTCOLOSSEO_PYTHON:-python}
task_output=reports/hierarchical/paired-runtime-20261006/videos
mkdir -p "$task_output"
trap 'task_status=$?; echo "render_exit_code=$task_status"' EXIT
for task_case in aggressive defensive; do
  task_seed=107
  task_role=opponent
  if [[ "$task_case" == defensive ]]; then task_seed=104; task_role=host; fi
  for task_arm in control switch; do
    task_args=()
    if [[ "$task_arm" == switch ]]; then
      task_args=(--switch --single-switch-style "$task_case")
    fi
    "$task_python" -u -m botcolosseo.cli.render_hierarchical \
      --executor runs/hierarchical/difficulty-anchored-pilot/epoch-3.pt \
      --strategy runs/hierarchical/random-switch-difficulty10k/last.pt \
      --opponent runs/hierarchical/difficulty-anchored-pilot/control-bundle/strategy-1.pt \
      --seed "$task_seed" --role "$task_role" --device cpu --difficulty 1 \
      --label "$task_case: $task_arm" --probe-style-response "${task_args[@]}" \
      --output "$task_output/$task_case-$task_arm.mp4"
  done
done
