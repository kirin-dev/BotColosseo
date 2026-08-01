#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]] || [[ ! "$1" =~ ^(defensive|explorer)$ ]]; then
  echo "Usage: $0 {defensive|explorer}" >&2
  exit 2
fi

STYLE="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-python}"
GPU="${BOTCOLOSSEO_GPU:-0}"
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"

SOURCE="${BOTCOLOSSEO_STYLE_SOURCE:-runs/extraction/styles/$STYLE}"
DESTINATION="runs/extraction/styles/$STYLE"
RANKING="$SOURCE/evaluation-v2/ranking.json"
BASE="$("$PYTHON" -m botcolosseo.cli.resolve_extraction_strong_artifact \
  --field checkpoint)"

readarray -t candidate_evidence < <("$PYTHON" - "$RANKING" "$STYLE" <<'PY'
import json
import sys

ranking = json.load(open(sys.argv[1], encoding="utf-8"))
selected = ranking.get("selected", {})
if (
    ranking.get("policy") != sys.argv[2]
    or ranking.get("selection_split") != "validation"
    or ranking.get("test_cases_accessed") is not False
    or not selected.get("checkpoint")
    or not selected.get("report")
):
    raise SystemExit("Style ranking identity does not match")
print(selected["checkpoint"])
print(selected["report"])
PY
)
CHECKPOINT="${candidate_evidence[0]}"
VALIDATION="${candidate_evidence[1]}"
TAG="$(basename "${CHECKPOINT%.pt}")"
HELDOUT="$SOURCE/evaluation-v2/$TAG-heldout.json"

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
    --policy "$STYLE" \
    --split heldout \
    --device cuda:0 \
    --output "$HELDOUT"
fi

if [[ -e "$DESTINATION/showcase.pt" || \
      -e "$DESTINATION/showcase-demonstration.json" ]]; then
  echo "Refusing to overwrite existing $STYLE Showcase artifact" >&2
  exit 1
fi

"$PYTHON" -u -m botcolosseo.cli.create_extraction_showcase_demonstration \
  --policy "$STYLE" \
  --checkpoint "$CHECKPOINT" \
  --validation-report "$VALIDATION" \
  --strong-validation-report "$STRONG_VALIDATION" \
  --heldout-report "$HELDOUT" \
  --strong-heldout-report "$STRONG_HELDOUT" \
  --output-checkpoint "$DESTINATION/showcase.pt" \
  --output-report "$DESTINATION/showcase-demonstration.json"

echo "Crystal Run: Extraction $STYLE validation demonstration complete"
