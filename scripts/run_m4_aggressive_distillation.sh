#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/home/wencong/miniconda3/envs/botcolosseo/bin/python
CONFIG="$ROOT/configs/m4/aggressive_distillation.yaml"
BASE="$ROOT/runs/m3/league-full/candidate-boundary-0200000.pt"
DATA="$ROOT/data/generated/m4/aggressive"
DISTILL_RUN="$ROOT/runs/m4/aggressive-distillation-main"
PPO_RUN="$ROOT/runs/m4/aggressive-distilled-main"
POOL="$ROOT/reports/m3/pools/pool-v1.json"
PAYOFFS="$ROOT/reports/m3/pools/payoffs-v1.json"

export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

if [[ -e "$DISTILL_RUN/metrics.jsonl" || -e "$PPO_RUN/metrics.jsonl" ]]; then
  echo "Refusing to overwrite an existing M4 distillation or PPO run" >&2
  exit 1
fi

mkdir -p "$DATA" "$DISTILL_RUN" "$PPO_RUN"

"$PYTHON" -u "$ROOT/scripts/generate_style_demonstrations.py" \
  --config "$CONFIG" \
  --output-dir "$DATA"

"$PYTHON" -u "$ROOT/scripts/train_style_distillation.py" \
  --config "$CONFIG" \
  --base-checkpoint "$BASE" \
  --run-dir "$DISTILL_RUN" \
  --device cuda:0

"$PYTHON" - "$DISTILL_RUN/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("completed") is not True:
    raise SystemExit("Style distillation did not complete")
if summary.get("offline_evaluation", {}).get("passed") is not True:
    raise SystemExit("Style distillation offline gate failed")
if summary.get("test_cases_accessed") is not False:
    raise SystemExit("Style distillation accessed test cases")
PY

"$PYTHON" -u "$ROOT/scripts/train_league.py" \
  --style aggressive \
  --style-warm-start "$DISTILL_RUN/style-pretrained.pt" \
  --config "$ROOT/configs/m4/aggressive.yaml" \
  --base-checkpoint "$BASE" \
  --pool "$POOL" \
  --payoffs "$PAYOFFS" \
  --run-dir "$PPO_RUN" \
  --device cuda:0
