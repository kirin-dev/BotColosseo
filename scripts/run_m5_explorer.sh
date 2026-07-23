#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/home/wencong/miniconda3/envs/botcolosseo/bin/python
CONFIG="$ROOT/configs/m5/explorer_distillation.yaml"
BASE="$ROOT/runs/m3/league-full/candidate-boundary-0200000.pt"
DATA="$ROOT/data/generated/m5/explorer"
DISTILL="$ROOT/runs/m5/explorer-distillation"
INTERPOLATION="$ROOT/runs/m5/explorer-interpolation"
SMOKE="$ROOT/reports/m5/explorer/smoke"
SELECTION="$ROOT/reports/m5/explorer/selection.json"
FORMAL="$ROOT/reports/m5/explorer/formal"

export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

if [[ -e "$DISTILL/metrics.jsonl" ]]; then
  echo "Refusing to overwrite an existing M5 Explorer distillation run" >&2
  exit 1
fi
if [[ -e "$FORMAL/manifest.json" ]]; then
  echo "Refusing to overwrite existing formal M5 Explorer evidence" >&2
  exit 1
fi

mkdir -p "$DATA" "$DISTILL" "$INTERPOLATION" "$SMOKE" "$(dirname "$SELECTION")"

if [[ ! -f "$DATA/train-manifest.json" ]]; then
  "$PYTHON" -u "$ROOT/scripts/generate_explorer_demonstrations.py" \
    --config "$CONFIG" \
    --base-checkpoint "$BASE" \
    --output-dir "$DATA" \
    --device cuda:0
else
  echo "Reusing existing hash-bound Explorer data manifest"
fi

"$PYTHON" -u "$ROOT/scripts/train_explorer_distillation.py" \
  --config "$CONFIG" \
  --base-checkpoint "$BASE" \
  --run-dir "$DISTILL" \
  --device cuda:0

for alpha in 0.25 0.50 0.75; do
  tag=$("$PYTHON" -c \
    'import sys; print(f"alpha-{round(float(sys.argv[1]) * 100):03d}")' "$alpha")
  "$PYTHON" -u "$ROOT/scripts/interpolate_explorer.py" \
    --neutral-checkpoint "$DISTILL/neutral.pt" \
    --distilled-checkpoint "$DISTILL/style-pretrained.pt" \
    --alpha "$alpha" \
    --output "$INTERPOLATION/$tag.pt" \
    --report "$INTERPOLATION/$tag.json"

  set +e
  "$PYTHON" -u "$ROOT/scripts/evaluate_explorer.py" \
    --base-checkpoint "$BASE" \
    --explorer-checkpoint "$INTERPOLATION/$tag.pt" \
    --output-dir "$SMOKE/$tag" \
    --pairs-per-opponent 1 \
    --max-decisions 525 \
    --max-attempts 2 \
    --bootstrap-samples 10000 \
    --bootstrap-seed 20260723 \
    --device cuda:0
  smoke_status=$?
  set -e
  if [[ "$smoke_status" -gt 1 ]]; then
    echo "Explorer smoke $tag failed before producing a gate result" >&2
    exit "$smoke_status"
  fi
done

"$PYTHON" -u "$ROOT/scripts/select_explorer.py" \
  --interpolation-dir "$INTERPOLATION" \
  --smoke-dir "$SMOKE" \
  --output "$SELECTION"

selected_checkpoint=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])' \
  "$SELECTION")

"$PYTHON" -u "$ROOT/scripts/evaluate_explorer.py" \
  --base-checkpoint "$BASE" \
  --explorer-checkpoint "$selected_checkpoint" \
  --output-dir "$FORMAL" \
  --pairs-per-opponent 10 \
  --max-decisions 525 \
  --max-attempts 2 \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260723 \
  --device cuda:0

"$PYTHON" -u "$ROOT/scripts/audit_explorer.py" --root "$ROOT"
