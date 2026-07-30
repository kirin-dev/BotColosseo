#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${BOTCOLOSSEO_PYTHON:-python}"
log_root="${repo_root}/runs/extraction-v2/aggressive-finisher-pipeline"
mkdir -p "${log_root}"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/src"

echo "[1/3] Train post-cache Extraction Finisher"
if [[ ! -f runs/extraction-v2/aggressive-finisher/summary.json ]]; then
  "${python_bin}" scripts/train_extraction_bc.py \
    --style strong \
    --device cuda:0 \
    --updates 2000 \
    --supervision-mode post-cache \
    --initial-checkpoint runs/extraction-v2/strong/best.pt \
    --train-manifest \
      data/generated/extraction-v2/aggressive/train/train-manifest.json \
    --validation-manifest \
      data/generated/extraction-v2/aggressive/validation/validation-manifest.json \
    --output-dir runs/extraction-v2/aggressive-finisher \
    >"${log_root}/train.log" 2>&1
fi

echo "[2/3] Evaluate frozen validation cases"
if [[ ! -f reports/extraction-v2/validation/aggressive-finisher.json ]]; then
  "${python_bin}" scripts/evaluate_extraction_policy.py \
    --checkpoint runs/extraction-v2/aggressive/best.pt \
    --base-checkpoint runs/extraction-v2/aggressive-finisher/best.pt \
    --aggressive-governor \
    --governor-carried 35 \
    --governor-health 1 \
    --governor-remaining 1 \
    --style aggressive \
    --split validation \
    --device cuda:0 \
    --output reports/extraction-v2/validation/aggressive-finisher.json \
    >"${log_root}/evaluate-validation.log" 2>&1
fi

echo "[3/3] Evaluate showcase candidate matrix"
if [[ ! -f reports/extraction-v2/showcase-candidates-aggressive-finisher.json ]]; then
  "${python_bin}" scripts/evaluate_extraction_policy.py \
    --checkpoint runs/extraction-v2/aggressive/best.pt \
    --base-checkpoint runs/extraction-v2/aggressive-finisher/best.pt \
    --aggressive-governor \
    --governor-carried 35 \
    --governor-health 1 \
    --governor-remaining 1 \
    --style aggressive \
    --cases configs/extraction_v2/showcase-candidates.json \
    --split validation \
    --device cuda:0 \
    --output reports/extraction-v2/showcase-candidates-aggressive-finisher.json \
    >"${log_root}/evaluate-showcase.log" 2>&1
fi
echo "Aggressive Extraction Finisher pipeline complete"
