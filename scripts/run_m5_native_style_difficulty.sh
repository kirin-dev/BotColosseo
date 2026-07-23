#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "defensive" && "$1" != "explorer" ) ]]; then
  echo "Usage: $0 defensive|explorer" >&2
  exit 2
fi

STYLE=$1
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/home/wencong/miniconda3/envs/botcolosseo/bin/python
BASE="$ROOT/runs/m3/league-full/candidate-boundary-0200000.pt"
CHECKPOINT="$ROOT/runs/m5/$STYLE-ppo-main/candidate-0200000.pt"
UPSTREAM="$ROOT/reports/m5/$STYLE/ppo-repair/formal/summary.json"
SMOKE="$ROOT/reports/m5/$STYLE/difficulty/smoke"
FORMAL="$ROOT/reports/m5/$STYLE/difficulty/formal"

export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "$CHECKPOINT" || ! -f "$UPSTREAM" ]]; then
  echo "Missing passing upstream $STYLE checkpoint or summary" >&2
  exit 1
fi
if [[ -e "$FORMAL/manifest.json" ]]; then
  echo "Refusing to overwrite formal $STYLE difficulty evidence" >&2
  exit 1
fi

"$PYTHON" - "$STYLE" "$CHECKPOINT" "$UPSTREAM" <<'PY'
import hashlib
import json
import sys

style, checkpoint_path, summary_path = sys.argv[1:]
summary = json.load(open(summary_path, encoding="utf-8"))
checkpoint_hash = hashlib.sha256(open(checkpoint_path, "rb").read()).hexdigest()
if (
    summary.get("stage") != f"m5-{style}"
    or summary.get("split") != "validation"
    or summary.get("passed") is not True
    or summary.get("complete") is not True
    or summary.get("test_cases_accessed") is not False
    or summary.get("checkpoint_sha256", {}).get(style) != checkpoint_hash
):
    raise SystemExit(f"{style} difficulty requires a passing formal style gate")
PY

mkdir -p "$SMOKE" "$FORMAL"

if [[ ! -f "$SMOKE/summary.json" ]]; then
  set +e
  "$PYTHON" -u "$ROOT/scripts/evaluate_native_style_difficulty.py" \
    --style "$STYLE" \
    --base-checkpoint "$BASE" \
    --style-checkpoint "$CHECKPOINT" \
    --output-dir "$SMOKE" \
    --pairs-per-opponent 1 \
    --max-decisions 525 \
    --max-attempts 2 \
    --bootstrap-samples 10000 \
    --bootstrap-seed 20260723 \
    --device cuda:0
  set -e
fi

"$PYTHON" - "$SMOKE/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    summary.get("complete") is not True
    or summary.get("episodes") != 60
    or summary.get("protocol_inconsistencies") != 0
    or summary.get("test_cases_accessed") is not False
):
    raise SystemExit("Native-style difficulty smoke is incomplete or inconsistent")
PY

"$PYTHON" -u "$ROOT/scripts/evaluate_native_style_difficulty.py" \
  --style "$STYLE" \
  --base-checkpoint "$BASE" \
  --style-checkpoint "$CHECKPOINT" \
  --output-dir "$FORMAL" \
  --pairs-per-opponent 10 \
  --max-decisions 525 \
  --max-attempts 2 \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260723 \
  --device cuda:0
