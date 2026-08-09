#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-python}"
GPU="${BOTCOLOSSEO_GPU:-0}"
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"

MEDIA="${BOTCOLOSSEO_SHOWCASE_MEDIA:-docs/assets/extraction}"
REPORTS="${BOTCOLOSSEO_SHOWCASE_REPORTS:-reports/extraction/showcase}"
METHOD="${BOTCOLOSSEO_SHOWCASE_METHOD:-docs/assets/extraction/method.svg}"
BASE="$("$PYTHON" -m botcolosseo.cli.resolve_extraction_strong_artifact \
  --field checkpoint)"
STRONG_MANIFEST="$("$PYTHON" \
  -m botcolosseo.cli.resolve_extraction_strong_artifact \
  --field manifest)"
strong_report="$("$PYTHON" \
  -m botcolosseo.cli.resolve_extraction_strong_artifact \
  --field validation_report)"
mkdir -p "$MEDIA" "$REPORTS"
declare -A style_checkpoint
declare -A style_report
declare -A style_manifest
for policy in aggressive defensive explorer; do
  resolved="$("$PYTHON" -m \
    botcolosseo.cli.resolve_extraction_showcase_artifact \
    --policy "$policy")"
  readarray -t artifact < <("$PYTHON" - "$resolved" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
print(payload["checkpoint"])
print(payload["validation_report"])
print(payload["manifest"])
PY
)
  style_checkpoint["$policy"]="${artifact[0]}"
  style_report["$policy"]="${artifact[1]}"
  style_manifest["$policy"]="${artifact[2]}"
done
aggressive_report="${style_report[aggressive]}"
defensive_report="${style_report[defensive]}"
explorer_report="${style_report[explorer]}"
selection="$REPORTS/selection.json"
if [[ ! -f "$selection" ]]; then
  "$PYTHON" -u -m botcolosseo.cli.select_extraction_showcases \
    --strong-report "$strong_report" \
    --strong-manifest "$STRONG_MANIFEST" \
    --aggressive-report "$aggressive_report" \
    --aggressive-manifest "${style_manifest[aggressive]}" \
    --defensive-report "$defensive_report" \
    --defensive-manifest "${style_manifest[defensive]}" \
    --explorer-report "$explorer_report" \
    --explorer-manifest "${style_manifest[explorer]}" \
    --output "$selection"
fi

for policy in strong aggressive defensive explorer; do
  case_index="$("$PYTHON" - "$selection" "$policy" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["selections"][sys.argv[2]]["case_index"])
PY
)"
  if [[ "$policy" == "strong" ]]; then
    checkpoint="$BASE"
    base_args=()
  else
    checkpoint="${style_checkpoint[$policy]}"
    base_args=(--base-checkpoint "$BASE")
  fi
  if [[ ! -f "$MEDIA/$policy.mp4" ]]; then
    "$PYTHON" -u -m botcolosseo.cli.render_extraction_v3 \
      --checkpoint "$checkpoint" \
      "${base_args[@]}" \
      --policy "$policy" \
      --case-index "$case_index" \
      --scenario-directory crystal_run_extraction_randomized \
      --device cuda:0 \
      --max-attempts 5 \
      --output "$MEDIA/$policy.mp4" \
      --evidence "$REPORTS/$policy.json"
  fi
done

if [[ ! -f "$MEDIA/showcase-board.png" ]]; then
  "$PYTHON" -u -m botcolosseo.cli.build_extraction_showcase_board \
    --strong-video "$MEDIA/strong.mp4" \
    --strong-evidence "$REPORTS/strong.json" \
    --aggressive-video "$MEDIA/aggressive.mp4" \
    --aggressive-evidence "$REPORTS/aggressive.json" \
    --defensive-video "$MEDIA/defensive.mp4" \
    --defensive-evidence "$REPORTS/defensive.json" \
    --explorer-video "$MEDIA/explorer.mp4" \
    --explorer-evidence "$REPORTS/explorer.json" \
    --selection "$selection" \
    --output "$MEDIA/showcase-board.png" \
    --manifest "$REPORTS/manifest.json"
fi

if [[ ! -f "$REPORTS/audit.json" ]]; then
  "$PYTHON" -u -m botcolosseo.cli.audit_extraction_showcase \
    --selection "$selection" \
    --board-manifest "$REPORTS/manifest.json" \
    --method "$METHOD" \
    --output "$REPORTS/audit.json"
fi

echo "Crystal Run: Extraction showcase complete"
