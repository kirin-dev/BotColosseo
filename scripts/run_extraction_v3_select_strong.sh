#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-python}"
GPU="${BOTCOLOSSEO_GPU:-0}"
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"

PPO_ROOT="runs/extraction/strong-ppo"
EVAL_ROOT="$PPO_ROOT/evaluation-v2"
mkdir -p "$EVAL_ROOT"

if [[ -f "$PPO_ROOT/selection.json" ]]; then
  if "$PYTHON" - "$PPO_ROOT/selection.json" <<'PY'
import json
import sys

selection = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(not (
    selection.get("gate_schema_version") == 2
    and selection.get("eligible") is True
    and selection.get("test_cases_accessed") is False
))
PY
  then
    echo "SKIP complete objective-aligned Strong selection"
    exit 0
  fi
  archive="$PPO_ROOT/archive/pre-objective-alignment"
  mkdir -p "$archive"
  for artifact in selection.json selected.pt; do
    if [[ -e "$PPO_ROOT/$artifact" ]]; then
      if [[ -e "$archive/$artifact" ]]; then
        echo "Refusing to overwrite archived legacy Strong $artifact" >&2
        exit 1
      fi
      mv "$PPO_ROOT/$artifact" "$archive/$artifact"
    fi
  done
  echo "Archived legacy Strong selection under $archive"
fi

legacy_ranking="$PPO_ROOT/evaluation/ranking.json"
ranking="$EVAL_ROOT/ranking.json"
if [[ -f "$legacy_ranking" ]]; then
  candidate_order="$legacy_ranking"
else
  reports=()
  for checkpoint in "$PPO_ROOT"/candidate-*.pt; do
    if [[ ! -f "$checkpoint" ]]; then
      echo "No Strong candidate checkpoints found" >&2
      exit 1
    fi
    selected_tag="$(basename "${checkpoint%.pt}")"
    validation_report="$EVAL_ROOT/$selected_tag-validation.json"
    if [[ ! -f "$validation_report" ]]; then
      "$PYTHON" -u -m botcolosseo.cli.evaluate_extraction_v3 \
        --checkpoint "$checkpoint" \
        --policy strong \
        --split validation \
        --device cuda:0 \
        --output "$validation_report"
    fi
    reports+=(--report "$validation_report")
  done
  if [[ ! -f "$ranking" ]]; then
    "$PYTHON" -u -m botcolosseo.cli.rank_extraction_candidates \
      --policy strong \
      "${reports[@]}" \
      --output "$ranking"
  fi
  candidate_order="$ranking"
fi

selected=false
while IFS= read -r selected_checkpoint; do
  selected_tag="$(basename "${selected_checkpoint%.pt}")"
  validation_report="$EVAL_ROOT/$selected_tag-validation.json"
  if [[ ! -f "$validation_report" ]]; then
    "$PYTHON" -u -m botcolosseo.cli.evaluate_extraction_v3 \
      --checkpoint "$selected_checkpoint" \
      --policy strong \
      --split validation \
      --device cuda:0 \
      --output "$validation_report"
  fi
  if "$PYTHON" - "$validation_report" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1], encoding="utf-8"))["metrics"]
worst = min(row["win_rate"] for row in metrics["by_opponent"].values())
raise SystemExit(not (
    metrics["win_rate"] >= 0.70
    and worst >= 0.55
    and metrics["extraction_rate"] >= 0.75
    and metrics["mean_extracted_value_advantage"] > 0
    and metrics["protocol_inconsistencies"] == 0
))
PY
  then
    heldout_report="$EVAL_ROOT/$selected_tag-heldout.json"
    solo_report="$EVAL_ROOT/$selected_tag-solo.json"
    full_gate_report="$EVAL_ROOT/$selected_tag-full-gate.json"
    if [[ ! -f "$heldout_report" ]]; then
      "$PYTHON" -u -m botcolosseo.cli.evaluate_extraction_v3 \
        --checkpoint "$selected_checkpoint" \
        --policy strong \
        --split heldout \
        --device cuda:0 \
        --output "$heldout_report"
    fi
    if [[ ! -f "$solo_report" ]]; then
      "$PYTHON" -u -m botcolosseo.cli.evaluate_extraction_v3 \
        --checkpoint "$selected_checkpoint" \
        --policy strong \
        --split solo \
        --device cuda:0 \
        --output "$solo_report"
    fi
    if "$PYTHON" -u -m botcolosseo.cli.select_extraction_candidate \
      --policy strong \
      --checkpoint "$selected_checkpoint" \
      --validation-report "$validation_report" \
      --heldout-report "$heldout_report" \
      --solo-report "$solo_report" \
      --output-checkpoint "$PPO_ROOT/selected.pt" \
      --output-report "$full_gate_report"
    then
      mv "$full_gate_report" "$PPO_ROOT/selection.json"
      selected=true
      break
    fi
  fi
done < <("$PYTHON" - "$candidate_order" <<'PY'
import json
import sys

ranking = json.load(open(sys.argv[1], encoding="utf-8"))
for candidate in sorted(
    ranking["candidates"],
    key=lambda item: tuple(item["score"]),
    reverse=True,
):
    print(candidate["checkpoint"])
PY
)
if [[ "$selected" != true ]]; then
  echo "No Strong candidate passed validation, heldout, and solo gates" >&2
  exit 2
fi

echo "Crystal Run: Extraction Strong selection complete"
