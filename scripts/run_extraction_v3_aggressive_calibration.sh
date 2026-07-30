#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-python}"
GPU="${BOTCOLOSSEO_GPU:-0}"
STOP_AFTER_STEPS="${BOTCOLOSSEO_STOP_AFTER_STEPS:-100000}"
if [[ ! "$STOP_AFTER_STEPS" =~ ^[1-9][0-9]*$ ]] || \
  (( STOP_AFTER_STEPS > 200000 )); then
  echo "BOTCOLOSSEO_STOP_AFTER_STEPS must be an integer in [1, 200000]" >&2
  exit 2
fi
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"

OUTPUT="runs/extraction/styles/aggressive-calibration-v2"
BASE="runs/extraction/strong-ppo/selected.pt"
PARENT="runs/extraction/styles/aggressive/candidate-0600000.pt"
for required in "$BASE" "$PARENT" "$(dirname "$PARENT")/summary.json"; do
  if [[ ! -f "$required" ]]; then
    echo "Aggressive calibration input is missing: $required" >&2
    exit 1
  fi
done

initialization=(--initialize-from "$PARENT")
if [[ -f "$OUTPUT/latest.pt" ]]; then
  initialization=(--resume "$OUTPUT/latest.pt")
fi
"$PYTHON" -u -m botcolosseo.cli.train_extraction_style \
  --config configs/extraction/styles-calibration.yaml \
  --style aggressive \
  --base-checkpoint "$BASE" \
  --device cuda:0 \
  --output-dir "$OUTPUT" \
  --stop-after-steps "$STOP_AFTER_STEPS" \
  "${initialization[@]}"

echo "Crystal Run: Extraction Aggressive calibration complete"
