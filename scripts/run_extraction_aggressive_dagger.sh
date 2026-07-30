#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${BOTCOLOSSEO_PYTHON:-python}"
log_root="${repo_root}/runs/extraction-v2/aggressive-dagger-pipeline"
mkdir -p "${log_root}"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/src"

generate_corrections() {
  local split="$1"
  local transitions="$2"
  local output_dir="data/generated/extraction-v2/aggressive-dagger/${split}"
  local manifest="${output_dir}/${split}-manifest.json"
  if [[ -f "${manifest}" ]]; then
    echo "SKIP corrections ${split}: manifest exists"
    return
  fi
  if [[ -d "${output_dir}" ]] && find "${output_dir}" -mindepth 1 -print -quit | grep -q .; then
    echo "ERROR partial correction directory: ${output_dir}" >&2
    return 1
  fi
  "${python_bin}" scripts/generate_extraction_corrections.py \
    --checkpoint runs/extraction-v2/aggressive/best.pt \
    --style aggressive \
    --split "${split}" \
    --transitions "${transitions}" \
    --device cuda:0 \
    --output-dir "${output_dir}" \
    >"${log_root}/generate-${split}.log" 2>&1
}

echo "[1/4] Collect Aggressive on-policy correction states"
generate_corrections train 20000
generate_corrections validation 4000

echo "[2/4] Fine-tune residual branch from the frozen candidate"
if [[ ! -f runs/extraction-v2/aggressive-dagger/summary.json ]]; then
  "${python_bin}" scripts/train_extraction_bc.py \
    --style aggressive \
    --device cuda:0 \
    --updates 2500 \
    --initial-checkpoint runs/extraction-v2/aggressive/best.pt \
    --base-checkpoint runs/extraction-v2/strong/best.pt \
    --train-manifest \
      data/generated/extraction-v2/aggressive-dagger/train/train-manifest.json \
    --validation-manifest \
      data/generated/extraction-v2/aggressive-dagger/validation/validation-manifest.json \
    --output-dir runs/extraction-v2/aggressive-dagger \
    >"${log_root}/train.log" 2>&1
fi

echo "[3/4] Evaluate frozen validation cases"
if [[ ! -f reports/extraction-v2/validation/aggressive-dagger.json ]]; then
  "${python_bin}" scripts/evaluate_extraction_policy.py \
    --checkpoint runs/extraction-v2/aggressive-dagger/best.pt \
    --style aggressive \
    --split validation \
    --device cuda:0 \
    --output reports/extraction-v2/validation/aggressive-dagger.json \
    >"${log_root}/evaluate-validation.log" 2>&1
fi

echo "[4/4] Evaluate showcase candidate matrix"
if [[ ! -f reports/extraction-v2/showcase-candidates-aggressive-dagger.json ]]; then
  "${python_bin}" scripts/evaluate_extraction_policy.py \
    --checkpoint runs/extraction-v2/aggressive-dagger/best.pt \
    --style aggressive \
    --cases configs/extraction_v2/showcase-candidates.json \
    --split validation \
    --device cuda:0 \
    --output reports/extraction-v2/showcase-candidates-aggressive-dagger.json \
    >"${log_root}/evaluate-showcase.log" 2>&1
fi
echo "Aggressive DAgger correction pipeline complete"
