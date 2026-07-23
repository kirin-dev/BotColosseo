#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-/home/wencong/miniconda3/envs/botcolosseo/bin/python}"
DEVICE="${BOTCOLOSSEO_DEVICE:-cuda:0}"

run_style() {
  local style="$1"
  local smoke="$ROOT/reports/m5/hybrid/$style/difficulty/smoke"
  local formal="$ROOT/reports/m5/hybrid/$style/difficulty/formal"

  PYTHONPATH="$ROOT/src" "$PYTHON" -u \
    "$ROOT/scripts/evaluate_hybrid_difficulty.py" \
    --style "$style" \
    --output-dir "$smoke" \
    --pairs-per-opponent 1 \
    --device "$DEVICE"
  test "$(jq -r .passed "$smoke/summary.json")" = "true"

  PYTHONPATH="$ROOT/src" "$PYTHON" -u \
    "$ROOT/scripts/evaluate_hybrid_difficulty.py" \
    --style "$style" \
    --output-dir "$formal" \
    --pairs-per-opponent 10 \
    --device "$DEVICE"
}

run_style defensive
run_style explorer
