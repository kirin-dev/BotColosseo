#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-/home/wencong/miniconda3/envs/botcolosseo/bin/python}"
GPU="${BOTCOLOSSEO_GPU:-0}"
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"

MEDIA="docs/assets/extraction"
REPORTS="reports/extraction/showcase"
BASE="runs/extraction/strong-ppo/selected.pt"
mkdir -p "$MEDIA" "$REPORTS"

report_from_selection() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json
import sys

selection = json.load(open(sys.argv[1], encoding="utf-8"))
matches = []
for path in selection["evidence"]:
    if not path.endswith("-validation.json"):
        continue
    report = json.load(open(path, encoding="utf-8"))
    if report.get("policy") == sys.argv[2]:
        matches.append(path)
if len(matches) != 1:
    raise SystemExit("Selection has no unique policy validation report")
print(matches[0])
PY
}

strong_report="$(report_from_selection runs/extraction/strong-ppo/selection.json strong)"
aggressive_report="$(report_from_selection runs/extraction/styles/aggressive/selection.json aggressive)"
defensive_report="$(report_from_selection runs/extraction/styles/defensive/selection.json defensive)"
explorer_report="$(report_from_selection runs/extraction/styles/explorer/selection.json explorer)"
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
