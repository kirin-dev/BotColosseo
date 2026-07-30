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

OUTPUT="runs/extraction/styles/defensive-calibration-v2"
BASE="runs/extraction/strong-ppo/selected.pt"
if [[ ! -f "$BASE" ]]; then
  echo "Selected Strong Base is missing: $BASE" >&2
  exit 1
fi

resume=()
if [[ -f "$OUTPUT/latest.pt" ]]; then
  resume=(--resume "$OUTPUT/latest.pt")
elif [[ -e "$OUTPUT/metrics.jsonl" || -e "$OUTPUT/summary.json" ]]; then
  echo "Refusing incomplete Defensive calibration without a checkpoint" >&2
  exit 1
fi
"$PYTHON" -u -m botcolosseo.cli.train_extraction_style \
  --config configs/extraction/styles-defensive-calibration.yaml \
  --style defensive \
  --base-checkpoint "$BASE" \
  --device cuda:0 \
  --output-dir "$OUTPUT" \
  --stop-after-steps "$STOP_AFTER_STEPS" \
  "${resume[@]}"

echo "Crystal Run: Extraction Defensive calibration complete"
