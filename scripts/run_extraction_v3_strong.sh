#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-$(command -v python)}"
GPU="${BOTCOLOSSEO_GPU:-0}"
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"

DATA_ROOT="data/generated/extraction/strong"
BC_ROOT="runs/extraction/strong-bc/strong"
PPO_ROOT="runs/extraction/strong-ppo"

generate_split() {
  local split="$1"
  local output="$DATA_ROOT/$split"
  local manifest="$output/$split-manifest.json"
  if [[ -f "$manifest" ]]; then
    echo "SKIP completed demonstration split: $split"
    return
  fi
  local resume=()
  if [[ -f "$output/progress.json" ]]; then
    resume=(--resume)
  fi
  "$PYTHON" -u -m botcolosseo.cli.generate_extraction_demonstrations \
    --config configs/extraction/demonstrations.yaml \
    --split "$split" \
    --style strong \
    --output-dir "$output" \
    "${resume[@]}"
}

generate_split train
generate_split validation

if [[ -f "$BC_ROOT/summary.json" ]] && \
  "$PYTHON" - "$BC_ROOT/summary.json" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(not (
    summary.get("completed") is True
    and summary.get("updates") == summary.get("target_updates") == 10_000
    and summary.get("test_cases_accessed") is False
))
PY
then
  echo "SKIP completed Strong BC"
else
  bc_resume=()
  if [[ -f "$BC_ROOT/latest.pt" ]]; then
    bc_resume=(--resume "$BC_ROOT/latest.pt")
  fi
  "$PYTHON" -u -m botcolosseo.cli.train_extraction_bc \
    --config configs/extraction/bc.yaml \
    --style strong \
    --device cuda:0 \
    --output-dir "$BC_ROOT" \
    "${bc_resume[@]}"
fi

if [[ -f "$PPO_ROOT/summary.json" ]] && \
  "$PYTHON" - "$PPO_ROOT/summary.json" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(not (
    summary.get("completed") is True
    and summary.get("environment_steps") == 2_000_000
    and summary.get("test_cases_accessed") is False
))
PY
then
  echo "SKIP completed Strong PPO"
else
  ppo_resume=()
  if [[ -f "$PPO_ROOT/latest.pt" ]]; then
    ppo_resume=(--resume "$PPO_ROOT/latest.pt")
  fi
  "$PYTHON" -u -m botcolosseo.cli.train_extraction_strong \
    --config configs/extraction/strong-ppo.yaml \
    --device cuda:0 \
    --output-dir "$PPO_ROOT" \
    "${ppo_resume[@]}"
fi

echo "Crystal Run: Extraction Strong pipeline complete"
