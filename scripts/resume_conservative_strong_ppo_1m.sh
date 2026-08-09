#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="${repo_root}/runs/extraction-randomized/strong-ppo-conservative-v2"
log_dir="${repo_root}/runs/extraction-randomized/logs"
checkpoint="${run_dir}/latest.pt"
summary="${run_dir}/summary.json"
pid_file="${run_dir}/train.pid"
log_file="${log_dir}/strong-ppo-conservative-v2-resume-1m.log"
python_bin="${BOTCOLOSSEO_PYTHON:-python}"

if [[ ! -f "${checkpoint}" || ! -f "${summary}" ]]; then
  echo "Conservative PPO v2 10k checkpoint is incomplete: ${run_dir}" >&2
  exit 1
fi
if [[ "$(jq -r '.environment_steps' "${summary}")" != "10000" ]]; then
  echo "Expected the admitted 10k checkpoint before the 1M resume" >&2
  exit 1
fi
if [[ -f "${pid_file}" ]]; then
  old_pid="$(tr -d '[:space:]' < "${pid_file}")"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Conservative PPO v2 is already running as PID ${old_pid}" >&2
    exit 1
  fi
fi
if [[ -e "${log_file}" ]]; then
  echo "Refusing to overwrite the 1M resume log: ${log_file}" >&2
  exit 1
fi

mkdir -p "${log_dir}"
cd "${repo_root}"
setsid nohup env \
  -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY CUDA_VISIBLE_DEVICES=1 \
  "${python_bin}" -u -m botcolosseo.cli.train_extraction_strong \
  --config configs/extraction/randomized/aligned-v2/strong-ppo-conservative.yaml \
  --device cuda:0 \
  --resume runs/extraction-randomized/strong-ppo-conservative-v2/latest.pt \
  --checkpoint-interval-steps 50000 \
  >"${log_file}" 2>&1 </dev/null &
pid=$!
disown "${pid}" 2>/dev/null || true
printf '%s\n' "${pid}" >"${pid_file}"
printf 'PID %s\nLOG %s\n' "${pid}" "${log_file}"
