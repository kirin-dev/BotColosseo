#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-python}"
GPU="${BOTCOLOSSEO_GPU:-0}"
TARGET_STEPS=51200
CONFIG="configs/extraction/randomized/styles-opportunity-pbrs-disengagement-v3.yaml"
OUTPUT="runs/extraction-randomized/styles-opportunity-pbrs-disengagement-v3/defensive"
CONTROL="runs/extraction-randomized/styles-opportunity-pbrs-disengagement-v3/control"
export PYTHONPATH="$ROOT/src"
cd "$ROOT"

[[ -x "$PYTHON" ]] || {
  echo "BotColosseo Python is not executable: $PYTHON" >&2
  exit 2
}
[[ -f "$CONFIG" ]] || {
  echo "Disengagement style config is missing: $CONFIG" >&2
  exit 2
}
mkdir -p "$CONTROL"

if [[ -f "$OUTPUT/summary.json" ]] && \
  "$PYTHON" - "$OUTPUT/summary.json" "$TARGET_STEPS" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(not (
    summary.get("style") == "defensive"
    and summary.get("environment_steps", 0) >= int(sys.argv[2])
    and summary.get("style_reward_schema_version") == 7
    and summary.get("opportunity_conditioning") is True
    and summary.get("frozen_strong_base") is True
    and summary.get("test_cases_accessed") is False
))
PY
then
  echo "SKIP completed defensive disengagement style"
  exit 0
fi

resume=()
if [[ -f "$OUTPUT/latest.pt" ]]; then
  resume=(--resume "$OUTPUT/latest.pt")
fi
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u \
  -m botcolosseo.cli.train_extraction_style \
  --config "$CONFIG" \
  --style defensive \
  --device cuda:0 \
  --output-dir "$OUTPUT" \
  --stop-after-steps "$TARGET_STEPS" \
  "${resume[@]}"
