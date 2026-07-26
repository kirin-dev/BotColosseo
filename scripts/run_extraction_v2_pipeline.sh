#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/wencong/miniconda3/envs/botcolosseo/bin/python"
log_root="${repo_root}/runs/extraction-v2/pipeline"
mkdir -p "${log_root}"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/src"

generate_split() {
  local style="$1"
  local split="$2"
  local output_dir="${repo_root}/data/generated/extraction-v2/${style}/${split}"
  local manifest="${output_dir}/${split}-manifest.json"
  if [[ -f "${manifest}" ]]; then
    echo "SKIP data ${style}/${split}: manifest exists"
    return
  fi
  if [[ -d "${output_dir}" ]] && find "${output_dir}" -mindepth 1 -print -quit | grep -q .; then
    echo "ERROR partial data directory requires inspection: ${output_dir}" >&2
    return 1
  fi
  "${python_bin}" scripts/generate_extraction_demonstrations.py \
    --split "${split}" \
    --style "${style}" \
    --output-dir "${output_dir}" \
    >"${log_root}/generate-${style}-${split}.log" 2>&1
}

train_policy() {
  local style="$1"
  local device="$2"
  local summary="${repo_root}/runs/extraction-v2/${style}/summary.json"
  if [[ -f "${summary}" ]]; then
    echo "SKIP train ${style}: summary exists"
    return
  fi
  "${python_bin}" scripts/train_extraction_bc.py \
    --style "${style}" \
    --device "${device}" \
    >"${log_root}/train-${style}.log" 2>&1
}

evaluate_policy() {
  local style="$1"
  local device="$2"
  local report="${repo_root}/reports/extraction-v2/validation/${style}.json"
  if [[ -f "${report}" ]]; then
    echo "SKIP evaluate ${style}: report exists"
    return
  fi
  "${python_bin}" scripts/evaluate_extraction_policy.py \
    --checkpoint "runs/extraction-v2/${style}/best.pt" \
    --style "${style}" \
    --split validation \
    --device "${device}" \
    --output "${report}" \
    >"${log_root}/evaluate-${style}.log" 2>&1
}

echo "[1/5] Generate training data"
data_pids=()
for style in strong aggressive defensive explorer; do
  generate_split "${style}" train &
  data_pids+=("$!")
done
for pid in "${data_pids[@]}"; do
  wait "${pid}"
done

echo "[2/5] Generate validation data"
data_pids=()
for style in strong aggressive defensive explorer; do
  generate_split "${style}" validation &
  data_pids+=("$!")
done
for pid in "${data_pids[@]}"; do
  wait "${pid}"
done

echo "[3/5] Train Strong Base"
train_policy strong cuda:0

echo "[4/5] Train residual style branches"
train_policy aggressive cuda:0 &
aggressive_pid="$!"
train_policy defensive cuda:1 &
defensive_pid="$!"
wait "${aggressive_pid}"
wait "${defensive_pid}"
train_policy explorer cuda:0

echo "[5/5] Run frozen validation and artifact audit"
evaluate_policy strong cuda:0
evaluate_policy aggressive cuda:0
evaluate_policy defensive cuda:1
evaluate_policy explorer cuda:1
if [[ ! -f reports/extraction-v2/training-artifact-audit.json ]]; then
  "${python_bin}" scripts/audit_extraction_artifacts.py \
    >"${log_root}/artifact-audit.log" 2>&1
fi
echo "Extraction v2 training pipeline complete"
