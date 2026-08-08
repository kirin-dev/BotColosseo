#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-python}"
TARGET_STEPS="${BOTCOLOSSEO_STYLE_STEPS:-100000}"
CONFIG="configs/extraction/randomized/styles-opportunity-pbrs.yaml"
OUTPUT_ROOT="runs/extraction-randomized/styles-opportunity-pbrs-v1"
CONTROL="$OUTPUT_ROOT/control"
export PYTHONPATH="$ROOT/src"
cd "$ROOT"

if [[ ! "$TARGET_STEPS" =~ ^[1-9][0-9]*$ ]] || (( TARGET_STEPS > 100000 )); then
  echo "BOTCOLOSSEO_STYLE_STEPS must be an integer in [1, 100000]" >&2
  exit 2
fi
[[ -x "$PYTHON" ]] || {
  echo "BotColosseo Python is not executable: $PYTHON" >&2
  exit 2
}
[[ -f "$CONFIG" ]] || {
  echo "Opportunity style config is missing: $CONFIG" >&2
  exit 2
}
mkdir -p "$CONTROL"

run_style() {
  local gpu="$1"
  local style="$2"
  local output="$OUTPUT_ROOT/$style"
  local log="$CONTROL/$style.log"
  local resume=()
  if [[ -f "$output/summary.json" ]] && \
    "$PYTHON" - "$output/summary.json" "$style" "$TARGET_STEPS" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(not (
    summary.get("style") == sys.argv[2]
    and summary.get("environment_steps", 0) >= int(sys.argv[3])
    and summary.get("opportunity_conditioning") is True
    and summary.get("frozen_strong_base") is True
    and summary.get("test_cases_accessed") is False
))
PY
  then
    printf 'SKIP completed %s at %s steps\n' "$style" "$TARGET_STEPS" >>"$log"
    return
  fi
  if [[ -f "$output/latest.pt" ]]; then
    resume=(--resume "$output/latest.pt")
  fi
  {
    printf '[%s] START %s on physical GPU %s\n' \
      "$(date --iso-8601=seconds)" "$style" "$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
      -m botcolosseo.cli.train_extraction_style \
      --config "$CONFIG" \
      --style "$style" \
      --device cuda:0 \
      --output-dir "$output" \
      --stop-after-steps "$TARGET_STEPS" \
      "${resume[@]}"
    printf '[%s] DONE %s\n' "$(date --iso-8601=seconds)" "$style"
  } >>"$log" 2>&1
}

run_lane() {
  local gpu="$1"
  shift
  for style in "$@"; do
    run_style "$gpu" "$style"
  done
}

printf '[%s] Starting opportunity-conditioned style lanes\n' \
  "$(date --iso-8601=seconds)"
run_lane 0 aggressive explorer &
lane0_pid=$!
run_lane 1 defensive &
lane1_pid=$!

lane0_status=0
lane1_status=0
wait "$lane0_pid" || lane0_status=$?
wait "$lane1_pid" || lane1_status=$?
printf '%s\n' "$lane0_status" >"$CONTROL/lane0.exit"
printf '%s\n' "$lane1_status" >"$CONTROL/lane1.exit"
if (( lane0_status != 0 || lane1_status != 0 )); then
  printf 'Style lane failure: GPU0=%s GPU1=%s\n' \
    "$lane0_status" "$lane1_status" >&2
  exit 1
fi
printf '[%s] Opportunity-conditioned style training complete\n' \
  "$(date --iso-8601=seconds)"
