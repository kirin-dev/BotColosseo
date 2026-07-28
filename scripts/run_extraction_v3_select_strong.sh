#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-/home/wencong/miniconda3/envs/botcolosseo/bin/python}"
GPU="${BOTCOLOSSEO_GPU:-0}"
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"

PPO_ROOT="runs/extraction/strong-ppo"
EVAL_ROOT="$PPO_ROOT/evaluation-v2"
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

if [[ ! -f "$PPO_ROOT/selection.json" ]]; then
  selected=false
  while IFS=$'\t' read -r selected_checkpoint validation_report; do
    selected_tag="$(basename "${selected_checkpoint%.pt}")"
    heldout_report="$EVAL_ROOT/$selected_tag-heldout.json"
    solo_report="$EVAL_ROOT/$selected_tag-solo.json"
    full_gate_report="$EVAL_ROOT/$selected_tag-full-gate.json"
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
    if "$PYTHON" -u -m botcolosseo.cli.select_extraction_candidate \
      --policy strong \
      --checkpoint "$selected_checkpoint" \
      --validation-report "$validation_report" \
      --heldout-report "$heldout_report" \
      --solo-report "$solo_report" \
      --output-checkpoint "$PPO_ROOT/selected.pt" \
      --output-report "$full_gate_report"
    then
      mv "$full_gate_report" "$PPO_ROOT/selection.json"
      selected=true
      break
    fi
  done < <("$PYTHON" - "$ranking" <<'PY'
import json
import sys

ranking = json.load(open(sys.argv[1], encoding="utf-8"))
for candidate in sorted(
    ranking["candidates"],
    key=lambda item: tuple(item["score"]),
    reverse=True,
):
    print(candidate["checkpoint"], candidate["report"], sep="\t")
PY
)
  if [[ "$selected" != true ]]; then
    echo "No Strong candidate passed validation, heldout, and solo gates" >&2
    exit 2
  fi
fi

echo "Crystal Run: Extraction Strong selection complete"
