#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]] || [[ ! "$1" =~ ^(aggressive|defensive|explorer)$ ]]; then
  echo "Usage: $0 {aggressive|defensive|explorer}" >&2
  exit 2
fi

STYLE="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-/home/wencong/miniconda3/envs/botcolosseo/bin/python}"
GPU="${BOTCOLOSSEO_GPU:-0}"
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"

OUTPUT="runs/extraction/styles/$STYLE"
BASE="runs/extraction/strong-ppo/selected.pt"
if [[ ! -f "$BASE" ]]; then
  echo "Selected Strong Base is missing: $BASE" >&2
  exit 1
fi
if [[ "$STYLE" != "aggressive" ]]; then
  AGGRESSIVE_SELECTION="runs/extraction/styles/aggressive/selection.json"
  if [[ ! -f "$AGGRESSIVE_SELECTION" ]] || \
    ! "$PYTHON" - "$AGGRESSIVE_SELECTION" <<'PY'
import json
import sys

selection = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(not (
    selection.get("policy") == "aggressive"
    and selection.get("gate_schema_version") == 2
    and selection.get("eligible") is True
    and selection.get("test_cases_accessed") is False
    and selection.get("selected_checkpoint_sha256")
))
PY
  then
    echo "Defensive/Explorer require a passing Aggressive vertical slice" >&2
    exit 1
  fi
fi

if [[ -f "$OUTPUT/summary.json" ]] && \
  "$PYTHON" - "$OUTPUT/summary.json" "$STYLE" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(not (
    summary.get("completed") is True
    and summary.get("environment_steps") == 600_000
    and summary.get("style") == sys.argv[2]
    and summary.get("frozen_strong_actor") is True
    and summary.get("frozen_strong_base") is True
    and summary.get("learned_residual_adapter") is True
    and summary.get("test_cases_accessed") is False
))
PY
then
  echo "SKIP completed style: $STYLE"
else
  resume=()
  if [[ -f "$OUTPUT/latest.pt" ]]; then
    resume=(--resume "$OUTPUT/latest.pt")
  fi
  "$PYTHON" -u -m botcolosseo.cli.train_extraction_style \
    --config configs/extraction/styles.yaml \
    --style "$STYLE" \
    --base-checkpoint "$BASE" \
    --device cuda:0 \
    --output-dir "$OUTPUT" \
    "${resume[@]}"
fi

echo "Crystal Run: Extraction $STYLE training complete"
