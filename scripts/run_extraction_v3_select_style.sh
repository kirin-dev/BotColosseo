#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]] || [[ ! "$1" =~ ^(aggressive|defensive|explorer)$ ]]; then
  echo "Usage: $0 {aggressive|defensive|explorer}" >&2
  exit 2
fi

STYLE="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-$(command -v python)}"
GPU="${BOTCOLOSSEO_GPU:-0}"
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"

OUTPUT="runs/extraction/styles/$STYLE"
EVAL_ROOT="$OUTPUT/evaluation"
BASE="runs/extraction/strong-ppo/selected.pt"
STRONG_RANKING="runs/extraction/strong-ppo/evaluation/ranking.json"
mkdir -p "$EVAL_ROOT"

strong_validation="$("$PYTHON" - "$STRONG_RANKING" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["report"])
PY
)"
reports=()
for checkpoint in "$OUTPUT"/candidate-*.pt; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "No $STYLE candidate checkpoints found" >&2
    exit 1
  fi
  tag="$(basename "${checkpoint%.pt}")"
  report="$EVAL_ROOT/$tag-validation.json"
  if [[ ! -f "$report" ]]; then
    "$PYTHON" -u -m botcolosseo.cli.evaluate_extraction_v3 \
      --checkpoint "$checkpoint" \
      --base-checkpoint "$BASE" \
      --policy "$STYLE" \
      --split validation \
      --device cuda:0 \
      --output "$report"
  fi
  reports+=(--report "$report")
done

ranking="$EVAL_ROOT/ranking.json"
if [[ ! -f "$ranking" ]]; then
  "$PYTHON" -u -m botcolosseo.cli.rank_extraction_candidates \
    --policy "$STYLE" \
    "${reports[@]}" \
    --strong-validation-report "$strong_validation" \
    --output "$ranking"
fi

selected_checkpoint="$("$PYTHON" - "$ranking" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["checkpoint"])
PY
)"
selected_report="$("$PYTHON" - "$ranking" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["report"])
PY
)"
if [[ ! -f "$OUTPUT/selection.json" ]]; then
  "$PYTHON" -u -m botcolosseo.cli.select_extraction_candidate \
    --policy "$STYLE" \
    --checkpoint "$selected_checkpoint" \
    --validation-report "$selected_report" \
    --strong-validation-report "$strong_validation" \
    --output-checkpoint "$OUTPUT/selected.pt" \
    --output-report "$OUTPUT/selection.json"
fi

echo "Crystal Run: Extraction $STYLE selection complete"
