#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="${repo_root}/runs/extraction-randomized/strong-ppo-1m"
log_dir="${repo_root}/runs/extraction-randomized/logs"
pid_file="${run_dir}/train.pid"
log_file="${log_dir}/strong-ppo-1m.log"
python_bin="${BOTCOLOSSEO_PYTHON:-python}"

if [[ -e "${run_dir}/metrics.jsonl" || -e "${pid_file}" ]]; then
  echo "Refusing to overwrite or duplicate the 1M run: ${run_dir}" >&2
  exit 1
fi

mkdir -p "${run_dir}" "${log_dir}"
cd "${repo_root}"
setsid nohup env \
  -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY CUDA_VISIBLE_DEVICES=0 \
  "${python_bin}" -m botcolosseo.cli.train_extraction_strong \
  --config configs/extraction/randomized/strong-ppo-1m.yaml \
  --device cuda:0 \
  >"${log_file}" 2>&1 </dev/null &
pid=$!
disown "${pid}" 2>/dev/null || true
printf '%s\n' "${pid}" >"${pid_file}"
printf 'PID %s\nLOG %s\n' "${pid}" "${log_file}"
