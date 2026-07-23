#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/home/wencong/miniconda3/envs/botcolosseo/bin/python
CONFIG="$ROOT/configs/m5/explorer_ppo.yaml"
BASE="$ROOT/runs/m3/league-full/candidate-boundary-0200000.pt"
WARM="$ROOT/runs/m5/explorer-interpolation/alpha-025.pt"
POOL="$ROOT/reports/m3/pools/pool-v1.json"
PAYOFFS="$ROOT/reports/m3/pools/payoffs-v1.json"
CUDA_SMOKE="$ROOT/runs/m5/explorer-ppo-cuda-smoke"
TRAIN="$ROOT/runs/m5/explorer-ppo-main"
EVAL_SMOKE="$ROOT/reports/m5/explorer/ppo-repair/smoke"
FORMAL="$ROOT/reports/m5/explorer/ppo-repair/formal"

export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

if [[ -e "$TRAIN/metrics.jsonl" || -e "$FORMAL/manifest.json" ]]; then
  echo "Refusing to overwrite Explorer PPO production evidence" >&2
  exit 1
fi

mkdir -p "$CUDA_SMOKE" "$TRAIN" "$EVAL_SMOKE" "$FORMAL"

if [[ ! -f "$CUDA_SMOKE/summary.json" ]]; then
  "$PYTHON" -u "$ROOT/scripts/train_league.py" \
    --style explorer \
    --style-warm-start "$WARM" \
    --config "$CONFIG" \
    --base-checkpoint "$BASE" \
    --pool "$POOL" \
    --payoffs "$PAYOFFS" \
    --run-dir "$CUDA_SMOKE" \
    --device cuda:0 \
    --stop-after-steps 2000
fi

"$PYTHON" - "$CUDA_SMOKE/summary.json" "$WARM" <<'PY'
import hashlib
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
warm_hash = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
if (
    summary.get("environment_steps") != 2000
    or summary.get("style") != "explorer"
    or summary.get("style_warm_start_sha256") != warm_hash
    or summary.get("test_cases_accessed") is not False
    or not summary.get("style_reward_components")
):
    raise SystemExit("Explorer PPO CUDA smoke identity or reward evidence is invalid")
PY

"$PYTHON" -u "$ROOT/scripts/train_league.py" \
  --style explorer \
  --style-warm-start "$WARM" \
  --config "$CONFIG" \
  --base-checkpoint "$BASE" \
  --pool "$POOL" \
  --payoffs "$PAYOFFS" \
  --run-dir "$TRAIN" \
  --device cuda:0

CANDIDATE="$TRAIN/candidate-0200000.pt"
"$PYTHON" -u "$ROOT/scripts/evaluate_explorer.py" \
  --base-checkpoint "$BASE" \
  --explorer-checkpoint "$CANDIDATE" \
  --output-dir "$EVAL_SMOKE" \
  --pairs-per-opponent 1 \
  --max-decisions 525 \
  --max-attempts 2 \
  --device cuda:0

"$PYTHON" -u "$ROOT/scripts/evaluate_explorer.py" \
  --base-checkpoint "$BASE" \
  --explorer-checkpoint "$CANDIDATE" \
  --output-dir "$FORMAL" \
  --pairs-per-opponent 10 \
  --max-decisions 525 \
  --max-attempts 2 \
  --device cuda:0

"$PYTHON" -u "$ROOT/scripts/audit_explorer_ppo.py" --root "$ROOT"
