#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]] || [[ ! "$1" =~ ^(aggressive|defensive|explorer)$ ]] || \
  [[ ! "$2" =~ ^[01]$ ]]; then
  echo "Usage: $0 {aggressive|defensive|explorer} {0|1}" >&2
  exit 2
fi

style="$1"
physical_gpu="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
selection="${repo_root}/reports/extraction/conservative-strong-1m-selection.json"
output_dir="${repo_root}/runs/extraction-randomized/styles-conservative-v2/${style}"
log_dir="${repo_root}/runs/extraction-randomized/logs"
pid_file="${output_dir}/train.pid"
log_file="${log_dir}/style-conservative-v2-${style}-200k.log"
python_bin="${BOTCOLOSSEO_PYTHON:-python}"

if [[ ! -f "${selection}" ]]; then
  echo "Conservative Strong selection has not completed: ${selection}" >&2
  exit 1
fi
base_checkpoint="$(jq -r '.selected_checkpoint' "${selection}")"
if [[ -z "${base_checkpoint}" || ! -f "${repo_root}/${base_checkpoint}" ]]; then
  echo "Selected Strong checkpoint is missing: ${base_checkpoint}" >&2
  exit 1
fi
if [[ -e "${output_dir}/metrics.jsonl" || -e "${pid_file}" || -e "${log_file}" ]]; then
  echo "Refusing to overwrite or duplicate ${style} style training" >&2
  exit 1
fi

mkdir -p "${output_dir}" "${log_dir}"
cd "${repo_root}"
setsid nohup env \
  -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY CUDA_VISIBLE_DEVICES="${physical_gpu}" \
  "${python_bin}" -u -m botcolosseo.cli.train_extraction_style \
  --config configs/extraction/randomized/styles-conservative.yaml \
  --style "${style}" \
  --base-checkpoint "${base_checkpoint}" \
  --device cuda:0 \
  --output-dir "runs/extraction-randomized/styles-conservative-v2/${style}" \
  --stop-after-steps 200000 \
  >"${log_file}" 2>&1 </dev/null &
pid=$!
disown "${pid}" 2>/dev/null || true
printf '%s\n' "${pid}" >"${pid_file}"
printf 'STYLE %s\nPID %s\nLOG %s\n' "${style}" "${pid}" "${log_file}"
