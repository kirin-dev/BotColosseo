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
BASE="${BOTCOLOSSEO_STRONG_CHECKPOINT:-runs/extraction-randomized/strong-ppo-conservative-v2/candidate-0950000.pt}"
STRONG_MANIFEST="${BOTCOLOSSEO_STRONG_MANIFEST:-reports/extraction/showcase/manifests/strong.json}"
strong_report="${BOTCOLOSSEO_STRONG_REPORT:-runs/extraction-randomized/strong-ppo-conservative-v2/evaluation-randomized-paired-style/candidate-0950000-validation.json}"
mkdir -p "$MEDIA" "$REPORTS"
declare -A style_checkpoint
declare -A style_report
declare -A style_manifest
style_checkpoint[aggressive]="${BOTCOLOSSEO_AGGRESSIVE_CHECKPOINT:-runs/extraction-randomized/styles-opportunity-pbrs-v1/aggressive/candidate-0051200.pt}"
style_checkpoint[defensive]="${BOTCOLOSSEO_DEFENSIVE_CHECKPOINT:-runs/extraction-randomized/styles-opportunity-pbrs-disengagement-v3/defensive/candidate-0051200.pt}"
style_checkpoint[explorer]="${BOTCOLOSSEO_EXPLORER_CHECKPOINT:-runs/extraction-randomized/styles-opportunity-pbrs-aligned-v2/explorer/candidate-0051200.pt}"
style_report[aggressive]="${BOTCOLOSSEO_AGGRESSIVE_REPORT:-runs/extraction-randomized/styles-opportunity-pbrs-v1/aggressive/evaluation-randomized/candidate-0051200-validation.json}"
style_report[defensive]="${BOTCOLOSSEO_DEFENSIVE_REPORT:-runs/extraction-randomized/styles-opportunity-pbrs-disengagement-v3/defensive/evaluation-randomized/candidate-0051200-validation.json}"
style_report[explorer]="${BOTCOLOSSEO_EXPLORER_REPORT:-runs/extraction-randomized/styles-opportunity-pbrs-aligned-v2/explorer/evaluation-randomized/candidate-0051200-validation.json}"
for policy in aggressive defensive explorer; do
  style_manifest["$policy"]="reports/extraction/showcase/manifests/$policy.json"
done
aggressive_report="${style_report[aggressive]}"
defensive_report="${style_report[defensive]}"
explorer_report="${style_report[explorer]}"
selection="$REPORTS/selection.json"
if [[ ! -f "$selection" ]]; then
  "$PYTHON" -u -m botcolosseo.cli.select_extraction_showcases \
    --strong-report "$strong_report" \
    --strong-manifest "$STRONG_MANIFEST" \
    --strong-case-index 201 \
    --aggressive-report "$aggressive_report" \
    --aggressive-manifest "${style_manifest[aggressive]}" \
    --aggressive-case-index 107 \
    --defensive-report "$defensive_report" \
    --defensive-manifest "${style_manifest[defensive]}" \
    --defensive-case-index 189 \
    --defensive-case-study \
    --explorer-report "$explorer_report" \
    --explorer-manifest "${style_manifest[explorer]}" \
    --explorer-case-index 20 \
    --protocol configs/extraction/randomized/evaluation.yaml \
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
