#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="${repo_root}/runs/extraction-randomized/strong-ppo-conservative-v2"
output_root="${repo_root}/reports/extraction/conservative-strong-1m-selection"
report="${repo_root}/reports/extraction/conservative-strong-1m-selection.json"
log_dir="${repo_root}/runs/extraction-randomized/logs"
pid_file="${output_root}/selection.pid"
log_file="${log_dir}/conservative-strong-1m-selection.log"
python_bin="${BOTCOLOSSEO_PYTHON:-python}"

if [[ ! -f "${run_dir}/summary.json" ]]; then
  echo "Conservative Strong 1M summary is missing: ${run_dir}" >&2
  exit 1
fi
if [[ -e "${pid_file}" || -e "${log_file}" || -e "${report}" ]]; then
  echo "Refusing to overwrite or duplicate Conservative Strong selection" >&2
  exit 1
fi

mkdir -p "${output_root}" "${log_dir}"
cd "${repo_root}"
setsid nohup env \
  -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY CUDA_VISIBLE_DEVICES=0,1 \
  "${python_bin}" -u -m botcolosseo.cli.select_randomized_strong_1m \
  --workers 2 \
  --run-dir runs/extraction-randomized/strong-ppo-conservative-v2 \
  --output-root reports/extraction/conservative-strong-1m-selection \
  --report reports/extraction/conservative-strong-1m-selection.json \
  >"${log_file}" 2>&1 </dev/null &
pid=$!
disown "${pid}" 2>/dev/null || true
printf '%s\n' "${pid}" >"${pid_file}"
printf 'PID %s\nLOG %s\n' "${pid}" "${log_file}"
