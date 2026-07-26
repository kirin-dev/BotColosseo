#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-$(command -v python)}"
GPU="${BOTCOLOSSEO_GPU:-0}"
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"

MEDIA="docs/assets/extraction"
REPORTS="reports/extraction/showcase"
BASE="runs/extraction/strong-ppo/selected.pt"
mkdir -p "$MEDIA" "$REPORTS"

report_from_ranking() {
  "$PYTHON" - "$1" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["report"])
PY
}

strong_report="$(report_from_ranking runs/extraction/strong-ppo/evaluation/ranking.json)"
aggressive_report="$(report_from_ranking runs/extraction/styles/aggressive/evaluation/ranking.json)"
defensive_report="$(report_from_ranking runs/extraction/styles/defensive/evaluation/ranking.json)"
explorer_report="$(report_from_ranking runs/extraction/styles/explorer/evaluation/ranking.json)"
selection="$REPORTS/selection.json"
if [[ ! -f "$selection" ]]; then
  "$PYTHON" -u -m botcolosseo.cli.select_extraction_showcases \
    --strong-report "$strong_report" \
    --aggressive-report "$aggressive_report" \
    --defensive-report "$defensive_report" \
    --explorer-report "$explorer_report" \
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
    checkpoint="runs/extraction/styles/$policy/selected.pt"
    base_args=(--base-checkpoint "$BASE")
  fi
  if [[ ! -f "$MEDIA/$policy.mp4" ]]; then
    "$PYTHON" -u -m botcolosseo.cli.render_extraction_v3 \
      --checkpoint "$checkpoint" \
      "${base_args[@]}" \
      --policy "$policy" \
      --case-index "$case_index" \
      --device cuda:0 \
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
    --output "$MEDIA/showcase-board.png" \
    --manifest "$REPORTS/manifest.json"
fi

echo "Crystal Run: Extraction showcase complete"
