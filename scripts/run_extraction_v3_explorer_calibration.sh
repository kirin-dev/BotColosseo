#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-python}"
GPU="${BOTCOLOSSEO_GPU:-0}"
STOP_AFTER_STEPS="${BOTCOLOSSEO_STOP_AFTER_STEPS:-200000}"
if [[ ! "$STOP_AFTER_STEPS" =~ ^(200000|400000)$ ]]; then
  echo "BOTCOLOSSEO_STOP_AFTER_STEPS must be 200000 or 400000" >&2
  exit 2
fi
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"

OUTPUT="runs/extraction/styles/explorer-calibration-v2"
BASE="$("$PYTHON" -m botcolosseo.cli.resolve_extraction_strong_artifact \
  --field checkpoint)"
resume=()
if [[ -f "$OUTPUT/latest.pt" ]]; then
  resume=(--resume "$OUTPUT/latest.pt")
elif [[ -e "$OUTPUT/metrics.jsonl" || -e "$OUTPUT/summary.json" ]]; then
  echo "Refusing incomplete Explorer calibration without a checkpoint" >&2
  exit 1
fi
"$PYTHON" -u -m botcolosseo.cli.train_extraction_style \
  --config configs/extraction/styles-explorer-calibration.yaml \
  --style explorer \
  --base-checkpoint "$BASE" \
  --device cuda:0 \
  --output-dir "$OUTPUT" \
  --stop-after-steps "$STOP_AFTER_STEPS" \
  "${resume[@]}"

echo "Crystal Run: Extraction Explorer calibration complete"
