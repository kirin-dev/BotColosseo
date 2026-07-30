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
BASE="runs/extraction/strong-ppo/selected.pt"
STRONG_SELECTION="runs/extraction/strong-ppo/selection.json"
if [[ ! -f "$RANKING" ]]; then
  echo "Style validation ranking is missing: $RANKING" >&2
  exit 1
fi

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

readarray -t strong_evidence < <("$PYTHON" - "$STRONG_SELECTION" <<'PY'
import json
import sys

selection = json.load(open(sys.argv[1], encoding="utf-8"))
if not (
    selection.get("policy") == "strong"
    and selection.get("gate_schema_version") == 2
    and selection.get("eligible") is True
    and selection.get("test_cases_accessed") is False
):
    raise SystemExit("Strong selection identity does not match")
for suffix in ("-validation.json", "-heldout.json"):
    matches = [path for path in selection["evidence"] if path.endswith(suffix)]
    if len(matches) != 1:
        raise SystemExit(f"Strong selection has no unique {suffix} evidence")
    print(matches[0])
PY
)
STRONG_VALIDATION="${strong_evidence[0]}"
STRONG_HELDOUT="${strong_evidence[1]}"

if [[ ! -f "$HELDOUT" ]]; then
  "$PYTHON" -u -m botcolosseo.cli.evaluate_extraction_v3 \
    --checkpoint "$CHECKPOINT" \
    --base-checkpoint "$BASE" \
    --policy "$STYLE" \
    --split heldout \
    --device cuda:0 \
    --output "$HELDOUT"
fi

if [[ -f "$DESTINATION/showcase-admission.json" || \
      -f "$DESTINATION/showcase.pt" ]]; then
  "$PYTHON" -m botcolosseo.cli.resolve_extraction_showcase_artifact \
    --policy "$STYLE"
  echo "SKIP valid $STYLE directional Showcase admission"
  exit 0
fi

"$PYTHON" -u -m botcolosseo.cli.admit_extraction_showcase \
  --policy "$STYLE" \
  --checkpoint "$CHECKPOINT" \
  --validation-report "$VALIDATION" \
  --strong-validation-report "$STRONG_VALIDATION" \
  --heldout-report "$HELDOUT" \
  --strong-heldout-report "$STRONG_HELDOUT" \
  --output-checkpoint "$DESTINATION/showcase.pt" \
  --output-report "$DESTINATION/showcase-admission.json"

echo "Crystal Run: Extraction $STYLE directional Showcase admission complete"
