#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${BOTCOLOSSEO_PYTHON:-/home/wencong/miniconda3/envs/botcolosseo/bin/python}
M2_WORKTREE=${M2_WORKTREE:-/home/wencong/BotColosseo/.worktrees/m2-base-training}
M3_PHYSICAL_GPU=${M3_PHYSICAL_GPU:-1}
RUN_DIR="$ROOT/runs/m3/league-full"
REPORT_ROOT="$ROOT/reports/m3"
POOL_DIR="$REPORT_ROOT/pools"
VALIDATION_DIR="$REPORT_ROOT/validation"
STATE_PATH="$RUN_DIR/pipeline-state.json"
BASE_CHECKPOINT="$ROOT/runs/m2/ppo-full/selected.pt"

export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="$M3_PHYSICAL_GPU"
export ROOT PYTHON RUN_DIR REPORT_ROOT POOL_DIR VALIDATION_DIR STATE_PATH
cd "$ROOT"

sync_artifact() {
  local source=$1
  local destination=$2
  mkdir -p "$(dirname "$destination")"
  if [[ -e "$destination" ]]; then
    if ! cmp -s "$source" "$destination"; then
      echo "Conflicting artifact: $destination" >&2
      return 1
    fi
    return 0
  fi
  install -m 0644 "$source" "$destination"
}

json_value() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in sys.argv[2].split("."):
    value = value[key]
print(str(value).lower() if isinstance(value, bool) else value)
PY
}

checkpoint_value() {
  "$PYTHON" - "$1" "$2" <<'PY'
import sys
from pathlib import Path
import torch

payload = torch.load(Path(sys.argv[1]), map_location="cpu", weights_only=False)
section, key = sys.argv[2].split(".")
print(payload[section][key])
PY
}

write_state() {
  local pool=$1
  local payoffs=$2
  local boundary=$3
  local mode=$4
  local checkpoint=$5
  "$PYTHON" - "$STATE_PATH" "$pool" "$payoffs" "$boundary" "$mode" "$checkpoint" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "schema_version": 1,
    "pool": sys.argv[2],
    "payoffs": sys.argv[3],
    "next_boundary": int(sys.argv[4]),
    "continuation_mode": sys.argv[5],
    "continuation_checkpoint": sys.argv[6],
}
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
with temporary.open("rb") as handle:
    os.fsync(handle.fileno())
os.replace(temporary, path)
PY
}

mkdir -p "$RUN_DIR" "$POOL_DIR" "$VALIDATION_DIR" "$ROOT/reports/m2"

for name in episodes.csv summary.json manifest.json; do
  sync_artifact "$M2_WORKTREE/reports/m2/$name" "$ROOT/reports/m2/$name"
done
sync_artifact \
  "$M2_WORKTREE/runs/m2/ppo-full/selected.pt" \
  "$BASE_CHECKPOINT"
sync_artifact \
  "$M2_WORKTREE/runs/m2/bc-full/best.pt" \
  "$ROOT/runs/m2/bc-full/best.pt"

QUALIFICATION_ARGS=()
if "$PYTHON" scripts/audit_m2_evidence.py --report-dir reports/m2 \
  >"$RUN_DIR/m2-strict-audit.json" 2>"$RUN_DIR/m2-strict-audit.err"; then
  echo "M2 capability audit: PASS"
else
  "$PYTHON" scripts/audit_m2_evidence.py \
    --report-dir reports/m2 \
    --integrity-only \
    >"$RUN_DIR/m2-integrity-audit.json"
  QUALIFICATION_ARGS=(--allow-integrity-qualified-base)
  echo "M2 capability audit: FAIL; integrity audit: PASS"
fi

if [[ ${M3_PREFLIGHT_ONLY:-0} == 1 ]]; then
  echo "M3 PIPELINE PREFLIGHT PASS"
  exit 0
fi

ANCHOR_VALIDATION="$ROOT/reports/m2/validation-anchor"
if [[ ! -f "$ANCHOR_VALIDATION/manifest.json" ]]; then
  if find "$ANCHOR_VALIDATION" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    echo "Partial M2 validation output exists: $ANCHOR_VALIDATION" >&2
    exit 1
  fi
  "$PYTHON" -u scripts/evaluate_m2.py \
    --split validation \
    --development \
    --policies ppo \
    --device cuda:0 \
    --output "$ANCHOR_VALIDATION"
fi

POOL="$POOL_DIR/pool-v0.json"
PAYOFFS="$POOL_DIR/payoffs-v0.json"
if [[ ! -f "$POOL" || ! -f "$PAYOFFS" ]]; then
  if [[ -e "$POOL" || -e "$PAYOFFS" ]]; then
    echo "Partial M3 bootstrap output exists" >&2
    exit 1
  fi
  "$PYTHON" scripts/bootstrap_m3_pool.py \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --validation-evidence-dir "$ANCHOR_VALIDATION" \
    --output-pool "$POOL" \
    --output-payoffs "$PAYOFFS" \
    --policy-id m2-anchor \
    --admitted-at-utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${QUALIFICATION_ARGS[@]}"
fi

BOUNDARY=200000
CONTINUATION_MODE=initial
CONTINUATION_CHECKPOINT=""
if [[ -f "$STATE_PATH" ]]; then
  POOL=$(json_value "$STATE_PATH" pool)
  PAYOFFS=$(json_value "$STATE_PATH" payoffs)
  BOUNDARY=$(json_value "$STATE_PATH" next_boundary)
  CONTINUATION_MODE=$(json_value "$STATE_PATH" continuation_mode)
  CONTINUATION_CHECKPOINT=$(json_value "$STATE_PATH" continuation_checkpoint)
else
  write_state "$POOL" "$PAYOFFS" "$BOUNDARY" initial ""
fi

while (( BOUNDARY <= 2000000 )); do
  LATEST="$RUN_DIR/latest.pt"
  CURRENT_POOL_HASH=$(json_value "$POOL" manifest_sha256)
  CURRENT_STEPS=0
  CURRENT_EPISODES=0
  if [[ -f "$LATEST" ]]; then
    LATEST_POOL_HASH=$(checkpoint_value "$LATEST" identity.pool_manifest_hash)
    CURRENT_STEPS=$(checkpoint_value "$LATEST" state.environment_steps)
    CURRENT_EPISODES=$(checkpoint_value "$LATEST" state.episodes)
    if [[ "$LATEST_POOL_HASH" == "$CURRENT_POOL_HASH" ]]; then
      CONTINUATION_MODE=resume
      CONTINUATION_CHECKPOINT="$LATEST"
    fi
  fi

  EXACT_BOUNDARY="$RUN_DIR/candidate-$(printf '%07d' "$BOUNDARY").pt"
  if (( BOUNDARY < 2000000 && CURRENT_STEPS > BOUNDARY && CURRENT_EPISODES % 2 != 0 )) \
    && [[ -f "$EXACT_BOUNDARY" ]]; then
    BOUNDARY_EPISODES=$(checkpoint_value "$EXACT_BOUNDARY" state.episodes)
    if (( BOUNDARY_EPISODES == CURRENT_EPISODES )); then
      RECOVERY_DIR="$RUN_DIR/recovery/unpaired-$BOUNDARY-$(date -u +%Y%m%dT%H%M%SZ)"
      mkdir -p "$RECOVERY_DIR"
      install -m 0644 "$LATEST" "$RECOVERY_DIR/latest.pt"
      install -m 0644 "$RUN_DIR/summary.json" "$RECOVERY_DIR/summary.json"
      install -m 0644 "$RUN_DIR/metrics.jsonl" "$RECOVERY_DIR/metrics.jsonl"
      echo "Archived incomplete cross-process padding at $RECOVERY_DIR"
      CONTINUATION_MODE=resume
      CONTINUATION_CHECKPOINT="$EXACT_BOUNDARY"
      CURRENT_STEPS=$(checkpoint_value "$EXACT_BOUNDARY" state.environment_steps)
      CURRENT_EPISODES=$BOUNDARY_EPISODES
    fi
  fi

  if (( CURRENT_STEPS < BOUNDARY || (BOUNDARY < 2000000 && CURRENT_EPISODES % 2 != 0) )); then
    TRAIN_ARGS=()
    if [[ "$CONTINUATION_MODE" == resume ]]; then
      TRAIN_ARGS=(--resume "$CONTINUATION_CHECKPOINT")
    elif [[ "$CONTINUATION_MODE" == transition ]]; then
      TRAIN_ARGS=(--transition-from "$CONTINUATION_CHECKPOINT")
    fi
    BOUNDARY_ARGS=()
    if (( BOUNDARY < 2000000 )); then
      BOUNDARY_ARGS=(--finish-paired-boundary)
    fi
    "$PYTHON" -u scripts/train_league.py \
      --config configs/m3/league.yaml \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --pool "$POOL" \
      --payoffs "$PAYOFFS" \
      --run-dir "$RUN_DIR" \
      --device cuda:0 \
      --stop-after-steps "$BOUNDARY" \
      "${BOUNDARY_ARGS[@]}" \
      "${QUALIFICATION_ARGS[@]}" \
      "${TRAIN_ARGS[@]}"
  fi

  CURRENT_STEPS=$(checkpoint_value "$LATEST" state.environment_steps)
  EPISODES=$(checkpoint_value "$LATEST" state.episodes)
  if (( BOUNDARY < 2000000 )); then
    if (( EPISODES % 2 != 0 )); then
      echo "Trainer returned before a paired episode boundary" >&2
      exit 1
    fi
  elif (( EPISODES % 2 != 0 )); then
    echo "Final 2M checkpoint is unpaired; selecting from earlier paired candidates"
    break
  fi

  CANDIDATE_ID=$(printf 'policy-%07d' "$CURRENT_STEPS")
  CANDIDATE="$RUN_DIR/candidate-boundary-$(printf '%07d' "$CURRENT_STEPS").pt"
  "$PYTHON" scripts/freeze_m3_candidate.py \
    --source "$LATEST" \
    --output "$CANDIDATE"

  CANDIDATE_DIR="$VALIDATION_DIR/$CANDIDATE_ID"
  MATRIX_DIR="$CANDIDATE_DIR/matrix"
  if [[ ! -f "$MATRIX_DIR/manifest.json" ]]; then
    if find "$MATRIX_DIR" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
      echo "Partial candidate matrix exists: $MATRIX_DIR" >&2
      exit 1
    fi
    "$PYTHON" -u scripts/evaluate_crossplay.py \
      --pool "$POOL" \
      --candidate-checkpoint "$CANDIDATE" \
      --candidate-id "$CANDIDATE_ID" \
      --include-scripts \
      --output-dir "$MATRIX_DIR" \
      --device cuda:0
  fi

  ENTRY="$CANDIDATE_DIR/entry.json"
  METRICS="$CANDIDATE_DIR/metrics.json"
  CANDIDATE_REPORT="$CANDIDATE_DIR/candidate-report.json"
  if [[ ! -f "$ENTRY" || ! -f "$METRICS" || ! -f "$CANDIDATE_REPORT" ]]; then
    if [[ -e "$ENTRY" || -e "$METRICS" || -e "$CANDIDATE_REPORT" ]]; then
      echo "Partial candidate admission preparation exists: $CANDIDATE_DIR" >&2
      exit 1
    fi
    "$PYTHON" scripts/prepare_candidate_admission.py \
      --pool "$POOL" \
      --candidate-checkpoint "$CANDIDATE" \
      --candidate-id "$CANDIDATE_ID" \
      --matrix-dir "$MATRIX_DIR" \
      --output-entry "$ENTRY" \
      --output-metrics "$METRICS" \
      --output-candidate-report "$CANDIDATE_REPORT" \
      --source-git-commit "$(git rev-parse HEAD)" \
      --admitted-at-utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  fi

  if (( BOUNDARY == 2000000 )); then
    break
  fi

  SOURCE_VERSION=$(json_value "$POOL" pool_version)
  NEXT_VERSION=$((SOURCE_VERSION + 1))
  NEXT_POOL="$POOL_DIR/pool-v$NEXT_VERSION.json"
  NEXT_PAYOFFS="$POOL_DIR/payoffs-v$NEXT_VERSION.json"
  DECISION="$CANDIDATE_DIR/admission-decision.json"
  if [[ ! -f "$DECISION" ]]; then
    set +e
    "$PYTHON" scripts/update_historical_pool.py \
      --pool "$POOL" \
      --entry "$ENTRY" \
      --metrics "$METRICS" \
      --output-pool "$NEXT_POOL" \
      --output-payoffs "$NEXT_PAYOFFS" \
      --decision-report "$DECISION" \
      --artifact-root "$ROOT"
    UPDATE_STATUS=$?
    set -e
    if [[ $UPDATE_STATUS -ne 0 && $UPDATE_STATUS -ne 2 ]]; then
      exit "$UPDATE_STATUS"
    fi
  fi

  if [[ $(json_value "$DECISION" eligible) == true ]]; then
    POOL="$NEXT_POOL"
    PAYOFFS="$NEXT_PAYOFFS"
    CONTINUATION_MODE=transition
  else
    CONTINUATION_MODE=resume
  fi
  CONTINUATION_CHECKPOINT="$CANDIDATE"
  BOUNDARY=$((BOUNDARY + 200000))
  write_state \
    "$POOL" "$PAYOFFS" "$BOUNDARY" \
    "$CONTINUATION_MODE" "$CONTINUATION_CHECKPOINT"
done

POOL_SIZE=$("$PYTHON" - "$POOL" <<'PY'
import json
import sys
from pathlib import Path
print(len(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["entries"]))
PY
)
if (( POOL_SIZE < 8 || POOL_SIZE > 12 )); then
  echo "M3 pool gate failed before test access: size=$POOL_SIZE" >&2
  exit 1
fi

mapfile -t CANDIDATE_REPORTS < <(
  find "$VALIDATION_DIR" -name candidate-report.json -type f | sort
)
if (( ${#CANDIDATE_REPORTS[@]} == 0 )); then
  echo "No M3 validation candidate reports exist" >&2
  exit 1
fi
SELECTION_ARGS=()
for report in "${CANDIDATE_REPORTS[@]}"; do
  SELECTION_ARGS+=(--candidate-report "$report")
done
SELECTION="$REPORT_ROOT/strong-base-selection.json"
if [[ ! -f "$SELECTION" ]]; then
  "$PYTHON" scripts/select_strong_base.py \
    "${SELECTION_ARGS[@]}" \
    --artifact-root "$ROOT" \
    --output "$SELECTION"
fi
SELECTED_CHECKPOINT="$ROOT/$(json_value "$SELECTION" selected.checkpoint)"

FINAL_CROSSPLAY="$REPORT_ROOT/final-crossplay"
if [[ ! -f "$FINAL_CROSSPLAY/manifest.json" ]]; then
  "$PYTHON" -u scripts/evaluate_crossplay.py \
    --pool "$POOL" \
    --output-dir "$FINAL_CROSSPLAY" \
    --device cuda:0
fi
sync_artifact "$FINAL_CROSSPLAY/crossplay.csv" "$REPORT_ROOT/crossplay.csv"

"$PYTHON" - "$POOL_DIR" "$REPORT_ROOT/pool-history.json" <<'PY'
import json
import os
import sys
from pathlib import Path
from botcolosseo.training.pfsp import pfsp_probabilities

pool_dir = Path(sys.argv[1])
output = Path(sys.argv[2])
snapshots = []
for pool_path in sorted(pool_dir.glob("pool-v*.json"), key=lambda p: int(p.stem.split("v")[-1])):
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    version = int(pool["pool_version"])
    payoff_path = pool_dir / f"payoffs-v{version}.json"
    payoffs = json.loads(payoff_path.read_text(encoding="utf-8"))["win_rates"]
    snapshots.append({
        "environment_steps": 0 if version == 0 else int(pool["entries"][-1]["environment_steps"]),
        "pool_size": len(pool["entries"]),
        "pfsp_probabilities": pfsp_probabilities(payoffs) if len(pool["entries"]) >= 2 else {},
    })
payload = {"schema_version": 1, "snapshots": snapshots}
temporary = output.with_name(f".{output.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
with temporary.open("rb") as handle:
    os.fsync(handle.fileno())
os.replace(temporary, output)
PY

OFFICIAL="$REPORT_ROOT/official"
if [[ ! -f "$OFFICIAL/episodes.jsonl" ]]; then
  "$PYTHON" scripts/evaluate_m3.py \
    --selection-report "$SELECTION" \
    --selected-checkpoint "$SELECTED_CHECKPOINT" \
    --pool "$POOL" \
    --m2-baseline "$BASE_CHECKPOINT" \
    --output-dir "$OFFICIAL" \
    --device cuda:0 \
    --preflight
fi

OFFICIAL_ARGS=()
if [[ -f "$OFFICIAL/episodes.jsonl" ]]; then
  OFFICIAL_ARGS=(--resume)
fi
"$PYTHON" -u scripts/evaluate_m3.py \
  --selection-report "$SELECTION" \
  --selected-checkpoint "$SELECTED_CHECKPOINT" \
  --pool "$POOL" \
  --m2-baseline "$BASE_CHECKPOINT" \
  --output-dir "$OFFICIAL" \
  --device cuda:0 \
  "${OFFICIAL_ARGS[@]}"

"$PYTHON" scripts/audit_m3_evidence.py --report-dir "$OFFICIAL"
"$PYTHON" scripts/render_m3_evidence.py \
  --official-report-dir "$OFFICIAL" \
  --crossplay-csv "$REPORT_ROOT/crossplay.csv" \
  --pool-history "$REPORT_ROOT/pool-history.json"

echo "M3 PIPELINE PASS"
