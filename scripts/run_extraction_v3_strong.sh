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
  local expected_transitions
  if [[ "$split" == "train" ]]; then
    expected_transitions=100000
  else
    expected_transitions=20000
  fi
  if [[ -f "$manifest" ]] && \
    "$PYTHON" - "$manifest" "$expected_transitions" <<'PY'
import json
import sys

from botcolosseo.data.extraction_demonstrations import extraction_teacher_sha256

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(not (
    manifest.get("transitions") == int(sys.argv[2])
    and manifest.get("target_transitions") == int(sys.argv[2])
    and manifest.get("max_decisions") == 700
    and manifest.get("shard_size") == 5000
    and manifest.get("teacher_implementation_sha256")
        == extraction_teacher_sha256()
    and manifest.get("test_cases_accessed") is False
))
PY
  then
    echo "SKIP completed demonstration split: $split"
    return
  fi
  if [[ -f "$manifest" ]]; then
    echo "REFUSE incompatible completed demonstration split: $split" >&2
    echo "Archive the existing $output before starting a new run." >&2
    exit 1
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
  "$PYTHON" - \
    "$BC_ROOT/summary.json" \
    "$DATA_ROOT/train/train-manifest.json" \
    "$DATA_ROOT/validation/validation-manifest.json" <<'PY'
import json
import sys
from pathlib import Path

from botcolosseo.data.demonstrations import sha256_file

summary = json.load(open(sys.argv[1], encoding="utf-8"))
train_manifest = Path(sys.argv[2])
validation_manifest = Path(sys.argv[3])
checkpoint = Path("runs/extraction/strong-bc/strong/best.pt")
raise SystemExit(not (
    summary.get("completed") is True
    and summary.get("updates") == summary.get("target_updates") == 10_000
    and summary.get("train_manifest_sha256") == sha256_file(train_manifest)
    and summary.get("validation_manifest_sha256")
        == sha256_file(validation_manifest)
    and summary.get("checkpoint_sha256") == sha256_file(checkpoint)
    and summary.get("test_cases_accessed") is False
))
PY
then
  echo "SKIP completed Strong BC"
else
  if [[ -f "$BC_ROOT/summary.json" ]]; then
    echo "REFUSE incompatible completed Strong BC" >&2
    echo "Archive the existing $BC_ROOT before starting a new run." >&2
    exit 1
  fi
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
  "$PYTHON" - "$PPO_ROOT/summary.json" "$BC_ROOT/best.pt" <<'PY'
import json
import sys
from pathlib import Path

from botcolosseo.data.demonstrations import sha256_file

summary = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(not (
    summary.get("completed") is True
    and summary.get("environment_steps") == 2_000_000
    and summary.get("bc_checkpoint_sha256") == sha256_file(Path(sys.argv[2]))
    and summary.get("freeze_actor_backbone") is True
    and summary.get("teacher_supervision")
        == "privileged-strong-training-only"
    and summary.get("test_cases_accessed") is False
))
PY
then
  echo "SKIP completed Strong PPO"
else
  if [[ -f "$PPO_ROOT/summary.json" ]]; then
    echo "REFUSE incompatible completed Strong PPO" >&2
    echo "Archive the existing $PPO_ROOT before starting a new run." >&2
    exit 1
  fi
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
