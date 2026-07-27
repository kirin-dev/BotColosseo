#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-$(command -v python)}"
GPU="${BOTCOLOSSEO_GPU:-0}"
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"

PPO_ROOT="runs/extraction/strong-ppo"
EVAL_ROOT="$PPO_ROOT/evaluation"
mkdir -p "$EVAL_ROOT"

reports=()
for checkpoint in "$PPO_ROOT"/candidate-*.pt; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "No Strong candidate checkpoints found" >&2
    exit 1
  fi
  tag="$(basename "${checkpoint%.pt}")"
  report="$EVAL_ROOT/$tag-validation.json"
  if [[ ! -f "$report" ]]; then
    "$PYTHON" -u -m botcolosseo.cli.evaluate_extraction_v3 \
      --checkpoint "$checkpoint" \
      --policy strong \
      --split validation \
      --device cuda:0 \
      --output "$report"
  fi
  reports+=(--report "$report")
done

ranking="$EVAL_ROOT/ranking.json"
if [[ ! -f "$ranking" ]]; then
  "$PYTHON" -u -m botcolosseo.cli.rank_extraction_candidates \
    --policy strong \
    "${reports[@]}" \
    --output "$ranking"
fi

selected_checkpoint="$("$PYTHON" - "$ranking" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["checkpoint"])
PY
)"
selected_tag="$(basename "${selected_checkpoint%.pt}")"
validation_report="$EVAL_ROOT/$selected_tag-validation.json"
heldout_report="$EVAL_ROOT/$selected_tag-heldout.json"
solo_report="$EVAL_ROOT/$selected_tag-solo.json"
if [[ ! -f "$heldout_report" ]]; then
  "$PYTHON" -u -m botcolosseo.cli.evaluate_extraction_v3 \
    --checkpoint "$selected_checkpoint" \
    --policy strong \
    --split heldout \
    --device cuda:0 \
    --output "$heldout_report"
fi
if [[ ! -f "$solo_report" ]]; then
  "$PYTHON" -u -m botcolosseo.cli.evaluate_extraction_v3 \
    --checkpoint "$selected_checkpoint" \
    --policy strong \
    --split solo \
    --device cuda:0 \
    --output "$solo_report"
fi

if [[ ! -f "$PPO_ROOT/selection.json" ]]; then
  "$PYTHON" -u -m botcolosseo.cli.select_extraction_candidate \
    --policy strong \
    --checkpoint "$selected_checkpoint" \
    --validation-report "$validation_report" \
    --heldout-report "$heldout_report" \
    --solo-report "$solo_report" \
    --output-checkpoint "$PPO_ROOT/selected.pt" \
    --output-report "$PPO_ROOT/selection.json"
fi

echo "Crystal Run: Extraction Strong selection complete"
