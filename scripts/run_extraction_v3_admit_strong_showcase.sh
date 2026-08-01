#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-python}"
export PYTHONPATH="$ROOT/src"
cd "$ROOT"

OUTPUT="runs/extraction/strong-ppo"
EVALUATION="$OUTPUT/evaluation-v2"
if [[ -f "$OUTPUT/showcase-admission.json" || -f "$OUTPUT/showcase.pt" ]]; then
  "$PYTHON" -m botcolosseo.cli.resolve_extraction_strong_artifact
  echo "SKIP valid Strong product Showcase admission"
  exit 0
fi

"$PYTHON" -u -m botcolosseo.cli.admit_extraction_strong_showcase \
  --ranking "$EVALUATION/ranking.json" \
  --evaluation-root "$EVALUATION" \
  --output-checkpoint "$OUTPUT/showcase.pt" \
  --output-report "$OUTPUT/showcase-admission.json"

echo "Crystal Run: Extraction Strong product Showcase admission complete"
