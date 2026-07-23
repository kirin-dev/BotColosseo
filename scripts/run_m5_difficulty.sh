#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/home/wencong/miniconda3/envs/botcolosseo/bin/python
BASE="$ROOT/runs/m3/league-full/candidate-boundary-0200000.pt"
AGGRESSIVE="$ROOT/runs/m4/aggressive-interpolation/alpha-025.pt"
CONFIG="$ROOT/configs/difficulty.yaml"
SMOKE="$ROOT/reports/m5/difficulty/smoke"
FORMAL="$ROOT/reports/m5/difficulty/formal"

export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -e "$FORMAL/manifest.json" ]]; then
  echo "Refusing to overwrite existing formal M5 difficulty evidence" >&2
  exit 1
fi

mkdir -p "$SMOKE" "$FORMAL"

set +e
"$PYTHON" -u "$ROOT/scripts/evaluate_difficulty.py" \
  --base-checkpoint "$BASE" \
  --aggressive-checkpoint "$AGGRESSIVE" \
  --config "$CONFIG" \
  --output-dir "$SMOKE" \
  --pairs-per-opponent 1 \
  --max-decisions 525 \
  --max-attempts 2 \
  --device cuda:0
smoke_status=$?
set -e
if [[ "$smoke_status" -gt 1 ]]; then
  echo "Difficulty smoke failed before producing a gate result" >&2
  exit "$smoke_status"
fi
"$PYTHON" - "$SMOKE/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    summary.get("complete") is not True
    or summary.get("episodes") != 60
    or summary.get("protocol_inconsistencies") != 0
):
    raise SystemExit("Difficulty smoke did not complete with a clean protocol")
PY

"$PYTHON" -u "$ROOT/scripts/evaluate_difficulty.py" \
  --base-checkpoint "$BASE" \
  --aggressive-checkpoint "$AGGRESSIVE" \
  --config "$CONFIG" \
  --output-dir "$FORMAL" \
  --pairs-per-opponent 10 \
  --max-decisions 525 \
  --max-attempts 2 \
  --device cuda:0

"$PYTHON" -u "$ROOT/scripts/audit_difficulty.py" --root "$ROOT"
