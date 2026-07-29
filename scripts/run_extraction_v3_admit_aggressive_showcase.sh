#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-/home/wencong/miniconda3/envs/botcolosseo/bin/python}"
GPU="${BOTCOLOSSEO_GPU:-0}"
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"

OUTPUT="runs/extraction/styles/aggressive"
CHECKPOINT="$OUTPUT/candidate-0600000.pt"
VALIDATION="$OUTPUT/evaluation-v2/candidate-0600000-validation.json"
HELDOUT="$OUTPUT/evaluation-v2/candidate-0600000-heldout.json"
STRONG_SELECTION="runs/extraction/strong-ppo/selection.json"
BASE="runs/extraction/strong-ppo/selected.pt"

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
