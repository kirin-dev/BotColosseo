#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-python}"
GPU="${BOTCOLOSSEO_GPU:-0}"
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"

OUTPUT="runs/extraction/styles/aggressive"
CHECKPOINT="$OUTPUT/candidate-0600000.pt"
VALIDATION="$OUTPUT/evaluation-v2/candidate-0600000-validation.json"
HELDOUT="$OUTPUT/evaluation-v2/candidate-0600000-heldout.json"
BASE="$("$PYTHON" -m botcolosseo.cli.resolve_extraction_strong_artifact \
  --field checkpoint)"
STRONG_VALIDATION="$("$PYTHON" \
  -m botcolosseo.cli.resolve_extraction_strong_artifact \
  --field validation_report)"
STRONG_HELDOUT="$("$PYTHON" \
  -m botcolosseo.cli.resolve_extraction_strong_artifact \
  --field heldout_report)"

if [[ ! -f "$HELDOUT" ]]; then
  "$PYTHON" -u -m botcolosseo.cli.evaluate_extraction_v3 \
    --checkpoint "$CHECKPOINT" \
    --base-checkpoint "$BASE" \
    --policy aggressive \
    --split heldout \
    --device cuda:0 \
    --output "$HELDOUT"
fi

if [[ -f "$OUTPUT/showcase-admission.json" || -f "$OUTPUT/showcase.pt" ]]; then
  "$PYTHON" -m botcolosseo.cli.check_extraction_aggressive_prerequisite
  echo "SKIP valid Aggressive directional Showcase admission"
  exit 0
fi

"$PYTHON" -u -m botcolosseo.cli.admit_extraction_showcase \
  --checkpoint "$CHECKPOINT" \
  --validation-report "$VALIDATION" \
  --strong-validation-report "$STRONG_VALIDATION" \
  --heldout-report "$HELDOUT" \
  --strong-heldout-report "$STRONG_HELDOUT" \
  --output-checkpoint "$OUTPUT/showcase.pt" \
  --output-report "$OUTPUT/showcase-admission.json"

echo "Crystal Run: Aggressive directional Showcase admission complete"
