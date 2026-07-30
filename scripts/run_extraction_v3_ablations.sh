#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-python}"
TARGET_STEPS=200000
BASE="runs/extraction/strong-ppo/selected.pt"
CONTROL="runs/extraction/ablations/control"
export PYTHONPATH="$ROOT/src"
cd "$ROOT"

config_for() {
  local variant="$1"
  local style="$2"
  if [[ "$style" == "defensive" ]]; then
    printf 'configs/extraction/ablations/%s-defensive.yaml\n' "$variant"
  else
    printf 'configs/extraction/ablations/%s.yaml\n' "$variant"
  fi
}

validate_inputs() {
  [[ -x "$PYTHON" ]] || {
    echo "Ablation Python is not executable: $PYTHON" >&2
    return 1
  }
  [[ -f "$BASE" ]] || {
    echo "Selected Strong checkpoint is missing: $BASE" >&2
    return 1
  }
  "$PYTHON" -m botcolosseo.cli.check_extraction_aggressive_prerequisite
  for variant in reward-only reward-plus-kl; do
    for style in aggressive defensive explorer; do
      local config
      config="$(config_for "$variant" "$style")"
      [[ -f "$config" ]] || {
        echo "Ablation config is missing: $config" >&2
        return 1
      }
    done
  done
  for path in \
    runs/extraction/styles/aggressive/candidate-0200000.pt \
    runs/extraction/styles/aggressive/evaluation-v2/candidate-0200000-validation.json \
    runs/extraction/styles/defensive-calibration-v2/candidate-0200000.pt \
    runs/extraction/styles/defensive-calibration-v2/evaluation-v2/candidate-0200000-validation.json \
    runs/extraction/styles/explorer/candidate-0200000.pt \
    runs/extraction/styles/explorer/evaluation-v2/candidate-0200000-validation.json
  do
    [[ -f "$path" ]] || {
      echo "Full-method 200k evidence is missing: $path" >&2
      return 1
    }
  done
}

run_case() {
  local gpu="$1"
  local variant="$2"
  local style="$3"
  local config output checkpoint report log
  config="$(config_for "$variant" "$style")"
  output="runs/extraction/ablations/$variant/$style"
  checkpoint="$output/candidate-0200000.pt"
  report="$output/evaluation/candidate-0200000-validation.json"
  log="$CONTROL/$variant-$style.log"
  mkdir -p "$output/evaluation" "$CONTROL"

  {
    printf '[%s] START %s %s on physical GPU %s\n' \
      "$(date --iso-8601=seconds)" "$variant" "$style" "$gpu"
    if [[ ! -f "$checkpoint" ]]; then
      resume=()
      if [[ -f "$output/latest.pt" ]]; then
        resume=(--resume "$output/latest.pt")
      fi
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
        -m botcolosseo.cli.train_extraction_style \
        --config "$config" \
        --style "$style" \
        --base-checkpoint "$BASE" \
        --device cuda:0 \
        --output-dir "$output" \
        --stop-after-steps "$TARGET_STEPS" \
        "${resume[@]}"
    else
      printf '[%s] SKIP existing checkpoint %s\n' \
        "$(date --iso-8601=seconds)" "$checkpoint"
    fi
    if [[ ! -f "$report" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
        -m botcolosseo.cli.evaluate_extraction_v3 \
        --checkpoint "$checkpoint" \
        --base-checkpoint "$BASE" \
        --policy "$style" \
        --split validation \
        --device cuda:0 \
        --output "$report"
    else
      printf '[%s] SKIP existing validation %s\n' \
        "$(date --iso-8601=seconds)" "$report"
    fi
    printf '[%s] DONE %s %s\n' \
      "$(date --iso-8601=seconds)" "$variant" "$style"
  } >>"$log" 2>&1
}

run_lane() {
  local gpu="$1"
  shift
  while [[ "$#" -gt 0 ]]; do
    run_case "$gpu" "$1" "$2"
    shift 2
  done
}

validate_inputs
if [[ "${BOTCOLOSSEO_ABLATION_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Extraction v3 ablation preflight: PASS"
  exit 0
fi

mkdir -p "$CONTROL"
printf '[%s] Starting two ablation GPU lanes\n' "$(date --iso-8601=seconds)"
run_lane 0 \
  reward-only aggressive \
  reward-only explorer \
  reward-plus-kl defensive &
lane0_pid=$!
run_lane 1 \
  reward-only defensive \
  reward-plus-kl aggressive \
  reward-plus-kl explorer &
lane1_pid=$!

lane0_status=0
lane1_status=0
wait "$lane0_pid" || lane0_status=$?
wait "$lane1_pid" || lane1_status=$?
if (( lane0_status != 0 || lane1_status != 0 )); then
  printf 'Ablation lane failure: GPU0=%s GPU1=%s\n' \
    "$lane0_status" "$lane1_status" >&2
  exit 1
fi
printf '[%s] Extraction v3 ablation pipeline complete\n' \
  "$(date --iso-8601=seconds)"
