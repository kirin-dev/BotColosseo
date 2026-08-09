#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-python}"
TARGET_STEPS=51200
CONFIG="configs/extraction/randomized/styles-opportunity-pbrs-aligned-v2.yaml"
OUTPUT_ROOT="runs/extraction-randomized/styles-opportunity-pbrs-aligned-v2"
CONTROL="$OUTPUT_ROOT/control"
export PYTHONPATH="$ROOT/src"
cd "$ROOT"

[[ -x "$PYTHON" ]] || {
  echo "BotColosseo Python is not executable: $PYTHON" >&2
  exit 2
}
[[ -f "$CONFIG" ]] || {
  echo "Aligned opportunity style config is missing: $CONFIG" >&2
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
    and summary.get("style_reward_schema_version") == 6
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

printf '[%s] Starting aligned opportunity-conditioned style lanes\n' \
  "$(date --iso-8601=seconds)"
run_style 0 defensive &
defensive_pid=$!
run_style 1 explorer &
explorer_pid=$!

defensive_status=0
explorer_status=0
wait "$defensive_pid" || defensive_status=$?
wait "$explorer_pid" || explorer_status=$?
printf '%s\n' "$defensive_status" >"$CONTROL/defensive.exit"
printf '%s\n' "$explorer_status" >"$CONTROL/explorer.exit"
if (( defensive_status != 0 || explorer_status != 0 )); then
  printf 'Style lane failure: defensive=%s explorer=%s\n' \
    "$defensive_status" "$explorer_status" >&2
  exit 1
fi
printf '[%s] Aligned opportunity-conditioned style training complete\n' \
  "$(date --iso-8601=seconds)"
