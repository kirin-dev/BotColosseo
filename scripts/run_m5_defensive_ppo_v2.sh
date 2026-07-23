#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/home/wencong/miniconda3/envs/botcolosseo/bin/python
STAGE=${1:-preflight}
CONFIG="$ROOT/configs/m5/defensive_ppo_v2.yaml"
BASE="$ROOT/runs/m3/league-full/candidate-boundary-0200000.pt"
WARM="$ROOT/runs/m5/defensive-interpolation/alpha-025.pt"
POOL="$ROOT/reports/m3/pools/pool-v1.json"
PAYOFFS="$ROOT/reports/m3/pools/payoffs-v1.json"
PREFLIGHT="$ROOT/runs/m5/v2/defensive/preflight-002000"
PILOT="$ROOT/runs/m5/v2/defensive/pilot"
SMOKE="$ROOT/reports/m5/v2/defensive/smoke-050000"

export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

train() {
  "$PYTHON" -u "$ROOT/scripts/train_league.py" \
    --style defensive \
    --style-warm-start "$WARM" \
    --config "$CONFIG" \
    --base-checkpoint "$BASE" \
    --pool "$POOL" \
    --payoffs "$PAYOFFS" \
    "$@"
}

case "$STAGE" in
  preflight)
    [[ ! -e "$PREFLIGHT/metrics.jsonl" ]] || {
      echo "Refusing to overwrite Defensive V2 preflight" >&2
      exit 1
    }
    train --run-dir "$PREFLIGHT" --device cuda:0 --stop-after-steps 2000
    "$PYTHON" "$ROOT/scripts/audit_m5_v2_training.py" \
      --run-dir "$PREFLIGHT" --style defensive --expected-steps 2000
    ;;
  pilot)
    "$PYTHON" "$ROOT/scripts/audit_m5_v2_training.py" \
      --run-dir "$PREFLIGHT" --style defensive --expected-steps 2000
    [[ ! -e "$PILOT/metrics.jsonl" ]] || {
      echo "Refusing to overwrite Defensive V2 pilot" >&2
      exit 1
    }
    train --run-dir "$PILOT" --device cuda:0 --stop-after-steps 50000
    "$PYTHON" "$ROOT/scripts/audit_m5_v2_training.py" \
      --run-dir "$PILOT" --style defensive --expected-steps 50000
    ;;
  smoke)
    [[ ! -e "$SMOKE/manifest.json" ]] || {
      echo "Refusing to overwrite Defensive V2 smoke" >&2
      exit 1
    }
    "$PYTHON" -u "$ROOT/scripts/evaluate_defensive.py" \
      --base-checkpoint "$BASE" \
      --defensive-checkpoint "$PILOT/candidate-0050000.pt" \
      --output-dir "$SMOKE" \
      --pairs-per-opponent 1 \
      --max-decisions 525 \
      --max-attempts 2 \
      --bootstrap-samples 10000 \
      --bootstrap-seed 20260723 \
      --device cuda:0
    ;;
  continue)
    train --run-dir "$PILOT" --device cuda:0 \
      --resume "$PILOT/latest.pt" --stop-after-steps 100000
    "$PYTHON" "$ROOT/scripts/audit_m5_v2_training.py" \
      --run-dir "$PILOT" --style defensive --expected-steps 100000
    ;;
  *)
    echo "Usage: $0 {preflight|pilot|smoke|continue}" >&2
    exit 2
    ;;
esac
